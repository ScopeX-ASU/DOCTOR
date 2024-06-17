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
import logging
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
from core.models.layers.utils import (
    CrosstalkScheduler,
    GlobalTemperatureScheduler,
    PhaseVariationScheduler,
    calculate_grad_hessian,
)

set_ms()
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
    accuracy_vector.append(accuracy.item())

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

        if configs.noise.PV_schedule == "low":
            mean_schedule_fn = lambda x: 0.0025 * x
            std_schedule_fn = lambda x: 0.004 * x + 0.002

        elif configs.noise.PV_schedule == "high":
            mean_schedule_fn = lambda x: 0.01 * x
            std_schedule_fn = lambda x: 0.005 * x + 0.005

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
            noise_scenario_src=configs.noise.noise_scenario_src,
            noise_scenario_tgt=configs.noise.noise_scenario_tgt,
            random_state=0,
            device=device,
        )
        if configs.noise.TD_schedule == "linear":
            # schedule = "Spatial"
            TD_schedule_fn = lambda x: x + 300
        elif configs.noise.TD_schedule == "cosine":
            # schedule = "perturbation"
            TD_schedule_fn = lambda x: (0.5 * np.cos(10 * x) + 0.5) + 300
        elif configs.noise.TD_schedule == "uneven":
            TD_schedule_fn = lambda x: x**3 + 300
        else:
            raise NotImplementedError

        global_temperature_scheduler = GlobalTemperatureScheduler(
            T_max=100000,
            schedule_fn=TD_schedule_fn,
            T0=300,
            lambda_res=lambda_res,
            L_list=2 * np.pi * radius_list,
            device=torch.device("cuda"),
        )
        crosstalk_scheduler = CrosstalkScheduler(
            Size=[4, 4, 8, 8],
            crosstalk_coupling_factor=configs.noise.crosstalk_factor,
            interv_h=configs.noise.inter_h,
            interv_v=configs.noise.inter_v,
        )

        model.set_noise_schedulers(
            scheduler_dict={
                "phase_variation_scheduler": phase_variation_scheduler,
                "global_temp_scheduler": global_temperature_scheduler,
                "crosstalk_scheduler": crosstalk_scheduler,
            }
        )
        model.step_noise_scheduler(T=configs.noise.delta_T)

        model.set_phase_variation(configs.noise.set_PV)
        if configs.noise.set_PV:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with phase variation..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        # set global temp drift
        model.set_global_temp_drift(configs.noise.set_GTD)
        if configs.noise.set_GTD:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with global temperature drift..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        model.set_crosstalk_noise(configs.noise.set_Crosstalk)
        if configs.noise.set_Crosstalk:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with crosstalk noise..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        def build_validate_callback(
            model,
            num_steps,
            step_vector=[],
            acc_vector=[],
            loss_vector=[],
            cycle_vector=[],
            interval=10,
        ):
            def validate_callback(step, cycle, loss):
                if step % interval == 0 or step == num_steps - 1:
                    step_vector.append(step)
                    cycle_vector.append(cycle)
                    loss_vector.append(loss)
                    validate(
                        model=model,
                        validation_loader=validation_loader,
                        epoch=-1,
                        criterion=criterion,
                        loss_vector=[],
                        accuracy_vector=acc_vector,
                        device=device,
                    )

            return validate_callback

        step_vector = []
        acc_vector = []
        loss_vector = []
        cycle_vector = []
        validate_callback_fn = build_validate_callback(
            model,
            configs.mapping.num_steps,
            step_vector=step_vector,
            acc_vector=acc_vector,
            loss_vector=loss_vector,
            cycle_vector=cycle_vector,
            interval=configs.mapping.validate_interval,
        )
        calculate_grad_hessian(
            model, train_loader, criterion, num_samples=10, device="cuda:0"
        )
        model.gen_weight_salience(mode=configs.mapping.salience_mode)
        # model.gen_sparsity_mask(sparsity=configs.mapping.sparsity, mode=configs.mapping.sparsity_mode)
        model.map_to_hardware(
            [
                1,
                configs.dataset.in_channel,
                configs.dataset.img_height,
                configs.dataset.img_width,
            ],
            lr=configs.mapping.lr,
            num_steps=configs.mapping.num_steps,
            stop_thres=configs.mapping.stop_thres,
            average_times=configs.mapping.average_times,
            criterion=configs.mapping.criterion,
            validation_callback=validate_callback_fn,
            sparsity=configs.mapping.sparsity,
            sparsity_mode=configs.mapping.sparsity_mode,
        )
        print("steps", step_vector)
        print("cycle", cycle_vector)
        print("loss", loss_vector)
        print("acc", acc_vector)

        lg.info("Validate converted pre-trained model (MODE = phase) with Mapping")
        validate(model, validation_loader, -1, criterion, [], [], device)

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
