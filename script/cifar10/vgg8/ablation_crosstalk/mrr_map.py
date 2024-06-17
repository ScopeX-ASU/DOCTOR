import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'Experiment/log/cifar10/vgg8/ablation_crosstalk'
script = 'train_map.py'
config_file = 'config/cifar10/vgg8/pm/Test_all_NI.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    scaling, crosstalk_factor, set_PV, set_GTD, set_crosstalk  = args
    for scaling_i in scaling:
        for crosstalk_factor_i in crosstalk_factor:
            with open(os.path.join(root, f'validation_cnn3_NA_{scaling_i}_{crosstalk_factor_i}.log'), 'w') as wfid:
                exp = [
                    f"--noise.inter_h={60 * scaling_i}",
                    f"--noise.inter_v={200 * scaling_i}",
                    f"--noise.crosstalk_factor={crosstalk_factor_i}",
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
        # 10000, #noise_std_std
        # ['low','high'], #noise intensity
        # ['edge','corner'],  # noise distribution source
        # ['corner','edge'],  # noise distribution target
        # 10000, #delta_T 
        # ['linear','cosine','uneven'],   #temp schedule
        [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5],    # scaling of interval
        [2e-1, 1.6e-1, 1.2e-1, 1e-1, 8.2e-2, 7e-2, 6e-2, 5e-2, 4e-2], #crosstalk_factor
        False, #set_PV
        False, #set_GTD
        True  #set_Crosstalk
    )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")