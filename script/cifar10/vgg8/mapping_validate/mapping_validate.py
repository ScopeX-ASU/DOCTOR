'''
Date: 2023-11-15 15:52:54
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-16 04:42:09
FilePath: /L2ight_Robust/script/cifar10/vgg8/mapping_validate/mapping_validate.py
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'Experiment/log/cifar10/vgg8/mapping_validate'
script = 'train_map.py'
config_file = 'config/cifar10/vgg8/pm/sparsity.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    sparsity, noise_std_std, PV_scheduler, noise_scenario_src, noise_scenario_tgt, delta_T, TD_schedule, crosstalk_factor, set_PV, set_GTD, set_crosstalk, nsteps, lr, average_times, criterion, validation_interval  = args
    for spar_i in sparsity:
        with open(os.path.join(root, f'mapping_validate_vgg8_{spar_i}_ns-{nsteps}_lr-{lr}_avg-{average_times}_cri-{criterion}_itvl-{validation_interval}.log'), 'w') as wfid:
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
                f"--mapping.sparsity={spar_i}",
                f"--mapping.lr={lr}",
                f"--mapping.average_times={average_times}",
                f"--mapping.criterion={criterion}",
                f"--mapping.validate_interval={validation_interval}",

            ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(
        [1.0],
        10000, #noise_std_std
        'high', #noise intensity
        'edge',  # noise distribution source
        'corner',  # noise distribution target
        10000, #delta_T 
        'linear',   #temp schedule
        0.1, #crosstalk_factor
        True, #set_PV
        True, #set_GTD
        True,  #set_Crosstalk,
        50, #nsteps
        2e-3, #lr
        1, #average_times
        'mae', #criterion
        2 #validation_interval
    )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")