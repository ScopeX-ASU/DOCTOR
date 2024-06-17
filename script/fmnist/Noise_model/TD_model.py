'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-11-07 11:51:17
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-07 11:51:18
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/Noise_Model/TD'
script = 'unitest/test_scheduler.py'
config_file = 'config/fmnist/cnn3_mrr/noise_model/model.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    test_TD, TD_Tmax, TD_scheduler = args
    with open(os.path.join(root, f'noise_model_TD_{TD_scheduler}_{TD_Tmax}.log'), 'w') as wfid:
        exp = [
            f"--noise.test_TD={test_TD}",
            f"--noise.TD_Tmax={TD_Tmax}",
            f"--noise.TD_scheduler={TD_scheduler}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment('TD')  # set experiments first

    # tasks = [(True, 100, 'cosine')]
    tasks = [(True, 100, 'linear')]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: TD_model Done.")