'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-11-14 16:17:31
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-11-14 17:09:27
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/cifar100/resnet18/pretrain"
script = 'train_pretrain.py'
config_file = 'config/cifar10/resnet18/pretrain.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    data_name, n_class, noise_std = args
    with open(os.path.join(root, f'pretrain_{data_name}_wn-{noise_std:.3f}.log'), 'w') as wfid:
        exp = [
            f"--dataset.name={data_name}",
            f"--dataset.n_class={n_class}",
            f"--noise.weight_noise_std={noise_std}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = [("cifar10", 10)] #

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
