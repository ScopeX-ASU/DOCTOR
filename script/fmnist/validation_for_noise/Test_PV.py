import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs


root = 'log/fmnist/cnn3_mrr/Validation/linear_corner'
script = 'train_map.py'
config_file = 'config/fmnist/cnn3_mrr/pm/Test_all_NI.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    noise_std_mean, noise_std_std, set_PV = args
    for noise_std_std_i in noise_std_std:
         with open(os.path.join(root, f'Validation_PV-{noise_std_std_i}.log'), 'w') as wfid:
            exp = [
                f"--noise.noise_std_mean={noise_std_mean}",
                f"--noise.noise_std_std={noise_std_std_i}",
                f"--noise.set_PV={set_PV}"
            ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(0.002, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], True)]
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")