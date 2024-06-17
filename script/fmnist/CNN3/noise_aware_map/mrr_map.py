import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'Experiment/log/fmnist/cnn3_mrr/Validation'
script = 'train_map.py'
config_file = 'config/fmnist/cnn3_mrr/pm/Test_all_NI_NA.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    noise_std_std, PV_scheduler, noise_scenario_src, noise_scenario_tgt, delta_T, TD_schedule, crosstalk_factor, set_PV, set_GTD, set_crosstalk  = args
    for PV_scheduler_i in PV_scheduler:
        for noise_scenario_src_i in noise_scenario_src:
            for noise_scenario_tgt_i in noise_scenario_tgt:
                for TD_schedule_i in TD_schedule:
                    with open(os.path.join(root, f'validation_cnn3_NA_{PV_scheduler_i}_{noise_scenario_src_i}_{TD_schedule_i}.log'), 'w') as wfid:
                        exp = [
                            f"--noise.noise_std_std={noise_std_std}",
                            f"--noise.PV_sheduler={PV_scheduler_i}",
                            f"--noise.noise_scenario_src={noise_scenario_src_i}",
                            f"--noise.noise_scenario_tgt={noise_scenario_tgt_i}",
                            f"--noise.delta_T={delta_T}",
                            f"--noise.TD_schedule={TD_schedule_i}",
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
        10000, #noise_std_std
        ['low','high'], #noise intensity
        ['edge','corner'],  # noise distribution source
        ['corner','edge'],  # noise distribution target
        10000, #delta_T 
        ['linear','cosine','uneven'],   #temp schedule
        8.2e-2, #crosstalk_factor
        True, #set_PV
        True, #set_GTD
        True  #set_Crosstalk
    )]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")