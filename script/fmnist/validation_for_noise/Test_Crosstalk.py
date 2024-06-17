import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/Validation'
script = 'train_map.py'
config_file = 'config/fmnist/cnn3_mrr/pm/Test_all_NI.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    crosstalk_factor, set_crosstalk = args
    with open(os.path.join(root, f'Validation_cnn3_mrr-CT-{crosstalk_factor}.log'), 'w') as wfid:
        exp = [
            f"--noise.crosstalk_factor={crosstalk_factor}",
            f"--noise.set_Crosstalk={set_crosstalk}"
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(4.8, True)]   #4.8 to make the weights variation ~5%
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")