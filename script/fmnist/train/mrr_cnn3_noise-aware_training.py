import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'log/fmnist/cnn3_mrr/pretrain'
script = 'train_pretrain.py'
config_file = 'config/fmnist/cnn3_mrr/pretrain_NA.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    n_epochs, n_epochs_NA = args
    with open(os.path.join(root, f'pretrain_cnn3_mrr-{n_epochs + n_epochs_NA}.log'), 'w') as wfid:
        exp = [
            f"--run.n_epochs={n_epochs}",
            f"--run.n_epochs_NA={n_epochs_NA}"
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(100, 20)] # assign pure-training epochs and noise-aware training epochs
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")