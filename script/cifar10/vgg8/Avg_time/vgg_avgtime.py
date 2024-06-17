import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = 'Experiment/Avg_time'
script = 'scan_avgtime.py'
config_file = 'config/cifar10/vgg8/pm/cali_avg_time.yml'
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = [
        'python3',
        script,
        config_file
    ]
    (sparsity, 
     avg_time, 
     noise_std_std, 
     PV_scheduler, 
     noise_scenario_src, 
     noise_scenario_tgt, 
     delta_T, 
     TD_schedule, 
     crosstalk_factor, 
     map_num_step, 
     lr, 
     criterion, 
     sparsity_mode, 
     salience_mode,
     validation_interval,
     stop_thres, 
     set_PV, 
     set_GTD, 
     set_crosstalk
    )  = args
    for avg_time_i in avg_time:
        with open(os.path.join(root, f"sparsity_vgg8_{sparsity}_ns-{map_num_step}_lr-{lr}_avg-{avg_time_i}_cri-{criterion}_itvl-{validation_interval}_sp-{sparsity_mode}_salience-{salience_mode}_stop-{stop_thres}.log"), "w") as wfid:
            exp = [
                f"--noise.noise_std_std={noise_std_std}",
                f"--noise.PV_sheduler={PV_scheduler}",
                f"--noise.noise_scenario_src={noise_scenario_src}",
                f"--noise.noise_scenario_tgt={noise_scenario_tgt}",
                f"--noise.delta_T={delta_T}",
                f"--noise.TD_schedule={TD_schedule}",
                f"--noise.crosstalk_factor={crosstalk_factor}",
                f"--noise.set_PV={set_PV}",
                f"--noise.set_GTD={set_GTD}",
                f"--noise.set_Crosstalk={set_crosstalk}"
                f"--mapping.num_steps={map_num_step}",
                f"--mapping.lr={lr}",
                f"--mapping.average_times={avg_time_i}",
                f"--mapping.criterion={criterion}",
                f"--mapping.validate_interval={validation_interval}",
                f"--mapping.sparsity_mode={sparsity_mode}",
                f"--mapping.salience_mode={salience_mode}",
                f"--mapping.sparsity={sparsity}",
                f"--mapping.stop_thres={stop_thres}",
                ]
            logger.info(f"running command {pres + exp}")
            subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (
            1.0,  #sparsity
            [1, 2, 3, 4, 5, 10, 15, 20],    #average time used to sampling to calibrate
            10000, #noise_std_std
            'high', #noise intensity
            'edge',  # noise distribution source
            'corner',  # noise distribution target
            10000, #delta_T 
            'linear',   #temp schedule
            1e-1, #crosstalk_factor
            100, #num_step
            0.003, #lr
            'mae', # criterion
            'IS', #sparsity mode
            'first_grad', # salience
            1, #validation_interval
            0.00382,
            True, #set_PV
            True, #set_GTD
            True  #set_Crosstalk
        )
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")