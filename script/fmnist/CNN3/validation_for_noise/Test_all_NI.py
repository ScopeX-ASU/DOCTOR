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
    noise_std_mean, noise_std_std, delta_T, crosstalk_factor, set_PV, set_GTD, set_crosstalk  = args
    for noise_std_std_i in noise_std_std:
        for delta_T_i in delta_T:
            with open(os.path.join(root, f'validation_cnn3_mrr-NA.log'), 'w') as wfid:
                exp = [
                    f"--noise.noise_std_std={noise_std_std_i}",
                    f"--noise.delta_T={delta_T_i}",
                    f"--noise.set_PV={set_PV}",
                    f"--noise.set_GTD={set_GTD}"
                ]
                logger.info(f"running command {pres + exp}")
                subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(
        [1, 100, 200, 300, 400, 500, 600, 700, 800, 900], #noise_std_mean
        [0, 10, 20], #noise_std_std
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #delta_T
        8.2e-2, #crosstalk_factor
        True, #set_PV
        True, #set_GTD
        True  #set_Crosstalk
    )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")