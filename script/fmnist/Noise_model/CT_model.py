import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/Noise_Model/CT'
script = 'unitest/test_scheduler.py'
config_file = 'config/fmnist/cnn3_mrr/noise_model/model.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    test_CT, inter_h, inter_v = args
    with open(os.path.join(root, f'noise_model_CT_{inter_h}_{inter_v}.log'), 'w') as wfid:
        exp = [
            f"--noise.test_CT={test_CT}",
            f"--noise.inter_h={inter_h}",
            f"--noise.inter_v={inter_v}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment('CT')  # set experiments first

    tasks = [(True, 60, 200)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp:  CT_model Done.")