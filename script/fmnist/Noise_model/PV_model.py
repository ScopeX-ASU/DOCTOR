import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/Noise_Model'
script = 'unitest/test_scheduler.py'
config_file = 'config/fmnist/cnn3_mrr/pm/Test_all_NI.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    test_PV, Tmax, PV_schedule = args
    with open(os.path.join(root, f'noise_model_PV_{PV_schedule}_{Tmax}.log'), 'w') as wfid:
        exp = [
            f"--noise.test_PV={test_PV}",
            f"--noise.Tmax={Tmax}",
            f"--noise.PV_schedule={PV_schedule}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment('PV')  # set experiments first

    tasks = [(True, 100, 'linear')]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp:  PV_model Done.")