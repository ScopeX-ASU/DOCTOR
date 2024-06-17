"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:07:39
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:07:39
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.plot import set_ms
from pyutils.torch_train import (
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.models.devices.mrr_configs import lambda_res, radius_list
from core.models.layers import AddDropMRRBlockConv2d, AddDropMRRBlockLinear
from core.models.layers.utils import (
    CrosstalkScheduler,
    GlobalTemperatureScheduler,
    PhaseVariationScheduler,
)

set_ms()
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        classify_loss = criterion(output, target)

        loss = classify_loss

        loss.backward()

        optimizer.step()
        step += 1

        if batch_idx % configs.run.log_interval == 0:
            lg.info(
                "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    classify_loss.data.item(),
                )
            )
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics(
        {"train_acc": accuracy.data.item(), "lr": get_learning_rate(optimizer)},
        step=epoch,
    )


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    counter_probe = 0
    # prob_frq = 100
    # prob_history = 0.
    # probe_error = 1e-3
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

            counter_probe = counter_probe + 1

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics(
        {"val_acc": accuracy.data.item(), "val_loss": val_loss}, step=epoch
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if configs.run.deterministic == True:
        set_torch_deterministic()

    model = builder.make_model(device)
    print(model)
    train_loader, validation_loader = builder.make_dataloader()
    criterion = builder.make_criterion().to(device)

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}_icalg-{configs.ic.alg}_icadapt-{configs.ic.adaptive}_icbest-{configs.ic.best_record}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "inbit": configs.quantize.input_bit,
            "wbit": configs.quantize.weight_bit,
            "init_lr": configs.optimizer.lr,
            "ic_alg": configs.ic.alg,
            "ic_adapt": configs.ic.adaptive,
            "ic_best_record": configs.ic.best_record,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )
    lg.info(
        f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
    )

    try:
        lg.info(configs)
        load_model(
            model,
            configs.checkpoint.restore_checkpoint,
            ignore_size_mismatch=int(configs.checkpoint.no_linear),
        )

        lg.info("Validate pre-trained model (MODE = weight)...")
        validate(model, validation_loader, -3, criterion, [], [], device)
        loss = 0
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            loss = criterion(output, target)
            loss.backward()

        model.backup_ideal_weights()

        model.set_phase_variation(configs.noise.set_PV)

        for layer in model.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                # Define phase variation schdule: Noise Intensity
                if configs.noise.PV_scheduler == "low":
                    mean_schedule_fn = lambda x: 0.01 * x
                    std_schedule_fn = lambda x: 0.01 * x + 0.001
                elif configs.noise.PV_scheduler == "high":
                    mean_schedule_fn = lambda x: 0.05 * x
                    std_schedule_fn = lambda x: 0.01 * x + 0.001

                # schedule = "linear"
                phase_variation_scheduler = PhaseVariationScheduler(
                    size=[4, 4, 8, 8],
                    T_max=100000,
                    mean_schedule_fn=mean_schedule_fn,
                    std_schedule_fn=std_schedule_fn,
                    smoothing_kernel_size=5,
                    smoothing_factor=0.05,
                    smoothing_mode="arch",
                    min_std=0.001,
                    momentum=0.9,
                    noise_scenario_src="edge",
                    noise_scenario_tgt="corner",
                    random_state=0,
                    device=device,
                )
                layer.phase_variation_scheduler = phase_variation_scheduler
                for _ in range(configs.noise.noise_std_std):
                    layer.phase_variation_scheduler.step()

        if configs.noise.set_PV:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with phase variation..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        # set global temp variation
        model.set_global_temp_drift(configs.noise.set_GTD)
        for layer in model.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                # Define Temp scheduler
                if configs.noise.TD_schedule == "linear":
                    schedule_fn = lambda x: 1 * x + 300
                elif configs.noise.TD_schedule == "cosine":
                    schedule_fn = lambda x: (0.5 * np.cos(10 * x) + 0.5) + 300
                elif configs.noise.TD_schedule == "uneven":
                    schedule_fn = lambda x: x**3 + 300
                else:
                    raise NotImplementedError

                global_temperature_scheduler = GlobalTemperatureScheduler(
                    T_max=100000,
                    schedule_fn=schedule_fn,
                    T0=300,
                    lambda_res=lambda_res,
                    L_list=2 * np.pi * radius_list,
                    device=torch.device("cuda"),
                )
                layer.global_temp_scheduler = global_temperature_scheduler
                layer.global_temp_scheduler.reset()

                for _ in range(configs.noise.delta_T):
                    layer.global_temp_scheduler.step()

                print(layer.global_temp_scheduler.get_global_temp())

        if configs.noise.set_GTD:
            print()
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with global temperature variation..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        phase_error, weight_error = [], []
        model.set_crosstalk_noise(configs.noise.set_Crosstalk)
        for layer in model.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                crosstalk_scheduler = CrosstalkScheduler(
                    # miniblock= layer.miniblock,
                    crosstalk_coupling_factor=configs.noise.crosstalk_factor,
                    interv_h=configs.noise.inter_h,
                    interv_v=configs.noise.inter_v,
                )
                layer.crosstalk_scheduler = crosstalk_scheduler

                size = layer.weight.shape
                layer.crosstalk_scheduler.get_crosstalk_matrix(layer.phase)

                layer.phase = torch.matmul(
                    layer.crosstalk_scheduler.crosstalk_mask.to(layer.device),
                    layer.phase.data.view(
                        layer.phase.shape[0], layer.phase.shape[1], -1
                    ).unsqueeze(-1),
                ).view(size)

                weight_error.append(
                    (layer.build_weight() - layer._ideal_weight).norm(p=1)
                    / layer._ideal_weight.norm(p=1)
                )
                phase_error.append(
                    (
                        layer.build_phase_from_weight(layer.build_weight())[0]
                        - layer._ideal_phase
                    ).norm(p=1)
                    / layer._ideal_phase.norm(p=1)
                )
        print(
            f"phase error is {sum(phase_error)/len(phase_error)}, weight error is {sum(weight_error)/len(weight_error)}"
        )

        if configs.noise.set_Crosstalk:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with crosstalk noise..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
