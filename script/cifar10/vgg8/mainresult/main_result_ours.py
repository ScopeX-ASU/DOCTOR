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

root = "Experiment/log/cifar10/vgg8/mainresult"
script = "eval.py"
config_file = "config/cifar100/resnet18_mrr/pm/sparsity.yml"
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
        remap_interval,
        ckpt,
        # New added
        W_thres, 
        T_thres,
        probe_interval,
        cool_time,
    ) = args
    with open(os.path.join(root, f"main_{algo}_res18_{sparsity}_ns-{nsteps}_lr-{lr}_avg-{average_times}_cri-{criterion}_itvl-{validation_interval}_sp-{sparsity_mode}_salience-{salience_mode}_stop-{stop_thres}_pv-{PV_schedule}-{noise_scenario_src}-{noise_scenario_tgt}_td-{TD_schedule}-{hotspot}_remapitvl-{remap_interval}.log"), "w") as wfid:
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
            f"--remapping.salience_mode={salience_mode}",
            f"--remapping.alg=LAP",
            f"--remapping.tolerance=100",
            f"--remapping.interval={remap_interval}",
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

    PV_config = {
        "PV.1": ("low", "edge", "corner"),
        "PV.2": ("high", "edge", "corner"),
                 }
    TD_config = {
        # "TD.1": ("linear", "uniform"),
        # "TD.2": ("cosine", "uniform"),
        # "TD.3": ("linear", "corner"),
        "TD.4": ("cosine", "corner"),
    }
    tasks = [
        (   
            "calib_remap",
            0.2,
            0,  # noise_std_std
            PV[0],  # noise intensity
            PV[1],  # noise distribution source
            PV[2],  # noise distribution target
            0,  # delta_T
            TD[0],  # temp schedule
            TD[1],  # hotspot
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
            0.0015,
            200, #remap_interval
            "./checkpoint/SparseBP_MRR_ResNet18_CIFAR100_wb-32_ib-32_ideal_acc-73.57_epoch-190.pt", # checkpoint
            # New added
            0.05,   #W_thres
            0.01,   #T_thres
            200,
            cool_time,   #cool_time
        )
        # for PV in PV_config.values() for TD in TD_config.values() for interval in [-1] + list(range(1000, 3000, 1000))
        # for PV in PV_config.values() for TD in TD_config.values() for interval in list(range(3000, 5000, 1000))
        # for PV in PV_config.values() for TD in TD_config.values() for interval in list(range(5000, 8000, 1000))
        for PV in PV_config.values() for TD in TD_config.values() for cool_time in [200]
        # for PV in PV_config.values() for TD in TD_config.values() for interval in [10000]
    ]

    with Pool(2) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
