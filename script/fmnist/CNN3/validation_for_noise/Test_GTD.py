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
    delta_T, set_GTD = args
    for delta_T_i in delta_T:
        with open(os.path.join(root, f'Validation_cnn3_mrr-GTD-{delta_T_i}.log'), 'w') as wfid:
            exp = [
                f"--noise.delta_T={delta_T_i}",
                f"--noise.set_GTD={set_GTD}"
            ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], True)]  # Assign temp drift, 300.1K to 301K
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")