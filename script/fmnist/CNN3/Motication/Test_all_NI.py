import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'Experiment/Motivation'
script = 'motivation.py'
config_file = 'config/fmnist/cnn3_mrr/pm/Test_all_NI.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    step, crosstalk_factor, set_PV, set_GTD, set_crosstalk  = args
    for step_i in step:
        # for delta_T_i in delta_T:
        with open(os.path.join(root, f'motivation_cnn3_mrr_{step_i}.log'), 'w') as wfid:
            exp = [
                f"--noise.noise_std_std={step_i}",
                f"--noise.delta_T={step_i}",
                f"--noise.crosstalk_factor={crosstalk_factor}",
                f"--noise.set_PV={set_PV}",
                f"--noise.set_GTD={set_GTD}",
                f"--noise.set_Crosstalk={set_crosstalk}"
            ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000], #noise_std_std
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #delta_T
        1e-1, #crosstalk_factor
        True, #set_PV
        True, #set_GTD
        True  #set_Crosstalk
    )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")