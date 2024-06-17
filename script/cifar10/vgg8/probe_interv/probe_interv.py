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

root = "Experiment/log/cifar10/vgg8/probe_interv"
script = "scan_interv.py"
config_file = "config/cifar10/vgg8/pm/sparsity.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    (   
        algo,
        sparsity,
        noise_std_std,
        PV_schedule,
        noise_scenario_src,
        noise_scenario_tgt,
        delta_T,
        TD_schedule,
        hotspot,
        crosstalk_factor,
        set_PV,
        set_GTD,
        set_crosstalk,
        nsteps,
        lr,
        average_times,
        criterion,
        validation_interval,
        sparsity_mode,
        salience_mode,
        stop_thres,
        ckpt,
        # New added
        W_thres, 
        T_thres,
        probe_interval,
        cool_time,
    ) = args

    with open(os.path.join(root, f"remap_{algo}_vgg8_{sparsity}_ns-{nsteps}_lr-{lr}_salience-{salience_mode}_pv-{PV_schedule}-{noise_scenario_src}-{noise_scenario_tgt}_td-{TD_schedule}-{hotspot}_cool_{cool_time}.log"), "w") as wfid:
        exp = [
            f"--noise.sparsity={sparsity}",
            f"--noise.noise_std_std={noise_std_std}",
            f"--noise.PV_schedule={PV_schedule}",
            f"--noise.noise_scenario_src={noise_scenario_src}",
            f"--noise.noise_scenario_tgt={noise_scenario_tgt}",
            f"--noise.delta_T={delta_T}",
            f"--noise.TD_schedule={TD_schedule}",
            f"--noise.TD_hotspot_mode={hotspot}",
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
            f"--mapping.sparsity={sparsity}",
            f"--mapping.stop_thres={stop_thres}",
            f"--run.algo={algo}",
            f"--run.repeat=1",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--mapping.onchiptrain.interval=500",
            f"--mapping.onchiptrain.n_epochs=1",
            # New added
            f'--mapping.W_thres={W_thres}',
            f'--mapping.T_thres={T_thres}',
            f'--mapping.probe_interval={probe_interval}',
            f'--mapping.cool_time={cool_time}',
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (   
            "ideal",
            0.2,
            0,  # noise_std_std
            "high",  # noise intensity
            "edge",  # noise distribution source
            "corner",  # noise distribution target
            0,  # delta_T
            "cosine",  # temp schedule
            "corner",  # hotspot
            1e-1,  # crosstalk_factor
            True,  # set_PV
            True,  # set_GTD
            True,  # set_Crosstalk
            20, #nsteps
            2e-3, #lr
            1, #average_times
            'mae', #criterion
            50, #validation_interval
            'IS', #sparsity_mode
            'first_grad', #salience_mode
            0.0015, #0.00382,
            "./checkpoint/SparseBP_MRR_VGG8_CIFAR10_wb-32_ib-32_ideal_acc-90.94_epoch-189.pt", # checkpoint
            # New added
            0.05,   #W_thres
            0.01,   #T_thres
            200,    #probe_interval,
            400,   #cool_time
        )

    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
