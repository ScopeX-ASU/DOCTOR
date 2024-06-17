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
import torch.nn.functional as F
from pyutils.config import configs

from core import builder
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
from core.models.layers import *
from core.models.layers.utils import *
from core.models.devices.mrr_configs import *

from pyutils.plot import plt, set_ms
import torch

set_ms()
import logging

logging.getLogger("matplotlib.font_manager").disabled = True
from pyutils.compute import merge_chunks
from pyutils.general import ensure_dir
import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
) -> None:
    is_train = model.training
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if batch_idx > len(train_loader) // 60: # 60k / 60 = 1000 image
            break # 10% calibration set

        optimizer.zero_grad()

        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        classify_loss = criterion(output, target)

        loss = classify_loss

        loss.backward()

        optimizer.step()
        step += 1
        model.step_noise_scheduler(T=3 * len(data)) # batch_size * forward + backward

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
    if scheduler is not None:
        scheduler.step()
    accuracy = 100.0 * correct.float() / (len(train_loader.dataset) // 60)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset) // 60} ({accuracy:.2f})%")
    mlflow.log_metrics(
        {"train_acc": accuracy.data.item(), "lr": get_learning_rate(optimizer)},
        step=epoch,
    )
    model.train(is_train)

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
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

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


def validate_noisy(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    enable_step: bool = True,
    algo: str = "ideal",
    repeat: int = 1,
    train_loader: DataLoader = None,
    optimizer: Optimizer = None,
    scheduler: Scheduler = None,
) -> None:
    model.eval()
    assert algo in ["ideal", "noise-aware", "onchiptrain", "calib", "calib_remap"]
    val_loss, correct = 0, 0
    cycles_per_inference = model.cycles(
        [
            1,
            configs.dataset.in_channel,
            configs.dataset.img_height,
            configs.dataset.img_width,
        ]
    )
    lg.info(f"cycles_per_inference: {cycles_per_inference}")
    total_cycles = 0
    step = 0
    counter = 0
    total_cali_cycles = 0
    model.backup_ideal_weights()
    
    for i in range(repeat):
        for datas, targets in tqdm.tqdm(validation_loader):
            datas = datas.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            for j in range(datas.shape[0]):
                data, target = datas[j : j + 1], targets[j : j + 1]
                with torch.no_grad():
                    output = model(data)

                val_loss += criterion(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
                model.step_noise_scheduler(1)
                total_cycles += cycles_per_inference
                if algo == "onchiptrain":
                    # model.set_enable_ste(True)
                    # model.set_noise_flag(True)   
                    if (step + 1) % configs.mapping.onchiptrain.interval == 0:
                        for ep in range(configs.mapping.onchiptrain.n_epochs):
                            train(
                                model=model,
                                train_loader=train_loader,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=ep,
                                criterion=criterion,
                                device=device,
                            )
                            ## noise scheduler update is inside of train()
                        n_inferences = int(configs.mapping.onchiptrain.n_epochs) * len(train_loader.dataset) * 3 // 60
                        total_cycles += n_inferences * cycles_per_inference
                        lg.info(f"cycles for onchiptrain: {n_inferences * cycles_per_inference}")
                elif algo == "remapping":
                    model.set_enable_remap(True)
                    if configs.remapping.interval > 0 and step % configs.remapping.interval == 0:
                        cycles = model.remap(
                            input_shape=[1, configs.dataset.in_channel, configs.dataset.img_height, configs.dataset.img_width],
                            flag=True,
                            alg=configs.remapping.alg,
                            salience_mode=configs.remapping.salience_mode,
                            average_times = 1,
                            tolerance=configs.remapping.tolerance,
                            verbose=True,
                            enable_step=True,
                        )
                        total_cycles += cycles

                elif algo == 'calib_remap':
                    model.set_enable_remap(True)
                    
                    remap_thres = configs.mapping.W_thres
                    T_thres = configs.mapping.T_thres
                    
                    cool_time = configs.mapping.cool_time    #configs.mapping.cool_time
                    # lg.info(f'cool_time = {cool_time}')
                    # lg.info(f'counter = {counter}')
                    if counter == 0 and (step % configs.remapping.interval == 0 or step == 10000 - 1) :
                        lg.info(f"step {step}, T: {model.global_temp_scheduler.T, model.global_temp_scheduler.T0} K")
                        # model.set_enable_remap(False)
                        # validate(model, validation_loader, -1, criterion, [], acc_vectors_noremap, device, average_times=3)
                        remap_flag = bool(((model.global_temp_scheduler.T - model.global_temp_scheduler._last_T) * 
                                            model.global_temp_scheduler.hotspot_map).mean() > T_thres)
                        lg.info(f'temperature flag is? ={remap_flag}')
                        if remap_flag is False:
                            remap_flag = (model.probe_weight_error() > remap_thres)
                            total_cycles += model.probe_cycles()

                        # model.recorded_probe_error = model.probe_weight_error()
                        # print(f'remap? = {remap_flag}, step = {step}')
                        lg.info(f'probed weight_noise is = {model.probe_weight_error()}')
                        lg.info(f'remap? ={remap_flag}')
                        if remap_flag:
                            remapping_cycles = model.remap(
                                input_shape=[1, configs.dataset.in_channel, configs.dataset.img_height, configs.dataset.img_width],
                                flag=True,
                                alg="LAP",
                                salience_mode="first_grad",
                                average_times = 1,
                                tolerance= 100,
                                verbose=True,
                                enable_step=True,
                            )
                            mapping_cycles = model.map_to_hardware(
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
                                # validation_callback=validate_callback_fn,
                                sparsity=configs.mapping.sparsity, 
                                sparsity_mode=configs.mapping.sparsity_mode
                            )
                            total_cali_cycles += mapping_cycles + remapping_cycles
                            total_cycles += mapping_cycles + remapping_cycles

                            counter = cool_time

                    counter = max(0, counter-1)

                step += 1

    val_loss /= len(validation_loader) * repeat
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset) / repeat
    accuracy_vector.append(accuracy.item())

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Cycles: {}\n".format(
            val_loss,
            correct,
            len(validation_loader.dataset) * repeat,
            accuracy,
            total_cycles,
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
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=configs.checkpoint.save_best_model_k)

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

    lossv, accv = [], []
    epoch = 0
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
            # mean_schedule_fn = lambda x: 0.0025 * x
            # std_schedule_fn = lambda x: 0.004 * x + 0.002
            mean_schedule_fn = lambda x: 0.005 * x
            std_schedule_fn = lambda x: 0.004 * x + 0.002

        elif configs.noise.PV_schedule == "high":
            # mean_schedule_fn = lambda x: 0.01 * x
            # std_schedule_fn = lambda x: 0.005 * x + 0.005
            mean_schedule_fn = lambda x: 0.025 * x
            std_schedule_fn = lambda x: 0.01 * x + 0.005

        phase_variation_scheduler = PhaseVariationScheduler(
            size=[4, 4, 8, 8],
            T_max=20000,
            # low noise : acc highest -> 89, 0.0025 * x, 0.0025 * x + 0.001
            # medium noise: acc highest -> 87,
            # high noise: acc highest -> 85, 0.005 * x, 0.002 * x + 0.001
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
            TD_schedule_fn = lambda x: (-0.25 * np.cos(10 * x) + 0.25) + 300
        elif configs.noise.TD_schedule == "uneven":
            TD_schedule_fn = lambda x: x**3 + 300
        else:
            raise NotImplementedError

        global_temperature_scheduler = GlobalTemperatureScheduler(
            size=[4, 4, 8, 8],
            T_max=20000,
            schedule_fn=TD_schedule_fn,
            T0=300,
            lambda_res=lambda_res,
            L_list=2 * np.pi * radius_list,
            hotspot_mode=getattr(configs.noise, "TD_hotspot_mode", "uniform"),
            device=torch.device("cuda"),
        )
        crosstalk_scheduler = CrosstalkScheduler(
            Size=[4, 4, 8, 8],
            crosstalk_coupling_factor=configs.noise.crosstalk_factor,
            interv_h=configs.noise.inter_h,
            interv_v=configs.noise.inter_v,
            device=device,
        )

        model.set_noise_schedulers(
            scheduler_dict={
                "phase_variation_scheduler": phase_variation_scheduler,
                "global_temp_scheduler": global_temperature_scheduler,
                "crosstalk_scheduler": crosstalk_scheduler,
            }
        )

        model.set_phase_variation(configs.noise.set_PV)
        if configs.noise.set_PV:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with phase variation..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        # set global temp variation
        model.set_global_temp_drift(configs.noise.set_GTD)
        if configs.noise.set_GTD:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with global temperature variation..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        model.set_crosstalk_noise(configs.noise.set_Crosstalk)

        if configs.noise.set_Crosstalk:
            lg.info(
                "Validate converted pre-trained model (MODE = phase) with crosstalk noise..."
            )
            validate(model, validation_loader, -1, criterion, [], [], device)

        # set all Non-ideality
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
        # model.map_to_hardware(
        #     [
        #         1,
        #         configs.dataset.in_channel,
        #         configs.dataset.img_height,
        #         configs.dataset.img_width,
        #     ],
        #     lr=configs.mapping.lr,
        #     num_steps=configs.mapping.num_steps,
        #     stop_thres=configs.mapping.stop_thres,
        #     average_times=configs.mapping.average_times,
        #     criterion=configs.mapping.criterion,
        #     validation_callback=validate_callback_fn,
        #     sparsity=configs.mapping.sparsity,
        #     sparsity_mode=configs.mapping.sparsity_mode
        # )
        # print("steps", step_vector)
        # print("cycle", cycle_vector)
        # print("loss", loss_vector)
        # print("acc", acc_vector)

        lg.info(
            f"Validate converted pre-trained model (MODE = phase) algo = {configs.run.algo}"
        )
        validate_noisy(
            model,
            validation_loader,
            -1,
            criterion,
            [],
            [],
            device,
            algo=configs.run.algo,
            repeat=configs.run.repeat,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=None,
        )

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
