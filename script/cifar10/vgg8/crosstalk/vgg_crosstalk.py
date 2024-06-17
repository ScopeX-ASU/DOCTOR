"""
Date: 2023-11-15 22:09:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-11-15 22:09:29
FilePath: /L2ight_Robust/script/cifar10/vgg8/Sparsity/vgg_sparsity.py
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "Experiment/log/cifar10/vgg8/scan_crosstalk"
script = "scan_crosstalk.py"
config_file = "config/cifar10/vgg8/pm/sparsity.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    (
        sparsity,
        noise_std_std,
        PV_scheduler,
        noise_scenario_src,
        noise_scenario_tgt,
        delta_T,
        TD_schedule,
        crosstalk_factor,
        set_PV,
        set_GTD,
        set_crosstalk,
        nsteps,
        lr,
        average_times,
        criterion,
        validation_interval,
        sparsity_modes,
        salience_modes,
        stop_thres
    ) = args
    for spar_i in sparsity:
        for sparsity_mode in sparsity_modes:
            for salience_mode in salience_modes:
                with open(os.path.join(root, f"crosstalk_vgg8_{spar_i}_ns-{nsteps}_lr-{lr}_avg-{average_times}_cri-{criterion}_itvl-{validation_interval}_sp-{sparsity_mode}_salience-{salience_mode}_stop-{stop_thres}.log"), "w") as wfid:
                    exp = [
                        f"--noise.sparsity={spar_i}",
                        f"--noise.noise_std_std={noise_std_std}",
                        f"--noise.PV_sheduler={PV_scheduler}",
                        f"--noise.noise_scenario_src={noise_scenario_src}",
                        f"--noise.noise_scenario_tgt={noise_scenario_tgt}",
                        f"--noise.delta_T={delta_T}",
                        f"--noise.TD_schedule={TD_schedule}",
                        f"--noise.crosstalk_factor={crosstalk_factor}",
                        f"--noise.set_PV={set_PV}",
                        f"--noise.set_GTD={set_GTD}",
                        f"--noise.set_Crosstalk={set_crosstalk}",
                        f"--mapping.num_steps={nsteps}",
                        f"--mapping.lr={lr}",
                        f"--mapping.average_times={average_times}",
                        f"--mapping.criterion={criterion}",
                        f"--mapping.validate_interval={validation_interval}",
                        f"--mapping.sparsity_mode={sparsity_mode}",
                        f"--mapping.salience_mode={salience_mode}",
                        f"--mapping.sparsity={spar_i}",
                        f"--mapping.stop_thres={stop_thres}",
                    ]
                    logger.info(f"running command {pres + exp}")
                    subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            [0.2],
            10000,  # noise_std_std
            "high",  # noise intensity
            "edge",  # noise distribution source
            "corner",  # noise distribution target
            10000,  # delta_T
            "linear",  # temp schedule
            1e-1,  # crosstalk_factor
            True,  # set_PV
            True,  # set_GTD
            True,  # set_Crosstalk
            200, #nsteps
            2e-3, #lr
            1, #average_times
            'mae', #criterion
            100, #validation_interval
            ['IS'], #sparsity_mode
            ['first_grad'], #salience_mode
            None,
        )
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
