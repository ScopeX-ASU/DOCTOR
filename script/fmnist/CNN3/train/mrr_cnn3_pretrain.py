'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-11-14 17:17:27
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-14 17:17:27
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/pretrain'
script = 'train_pretrain.py'
config_file = 'config/fmnist/cnn3_mrr/pretrain.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    n_epochs, noise_std = args
    with open(os.path.join(root, f'pretrain_cnn3_mrr-{n_epochs}_wn-{noise_std:.3f}.log'), 'w') as wfid:
        exp = [
            f"--run.n_epochs={n_epochs}",
            f"--noise.weight_noise_std={noise_std}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (100, 0.0),
        (100, 0.01)
        ]  # assign training epochs
    with Pool(2) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")