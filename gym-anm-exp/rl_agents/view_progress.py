"""
This script can be used to display some statistics about the training
status of the agents you are training.

Examples
--------
    $ python -m rl_agents.view_progress

Notes
-----
The above example will use the <BASE_DIR> and <ENV_ID> variables from
`hyperparameters.py` to construct the path to the trained agents.
"""
import os
import numpy as np
from .hyperparameters import BASE_DIR, ENV_ID
from .utils import get_monitor_elapsed_time

FOLDER = os.path.join(BASE_DIR, ENV_ID)
run_folders = sorted([f for f in os.listdir(FOLDER) if f.startswith('run')])
print('Run folders:', run_folders)

successful_runs = 0
incomplete_runs = 0

for run_folder in run_folders:
    if run_folder[:3] != 'run':
        continue
    
    run_id = run_folder.split('_')[-1]
    
    # Check if evaluations file exists
    eval_path = os.path.join(FOLDER, run_folder, 'evaluations.npz')
    if not os.path.exists(eval_path):
        print(f'RUN_ID: {run_id}, evaluations.npz not found (training not started or incomplete)')
        incomplete_runs += 1
        continue
    
    try:
        # Load last evaluation results.
        data = np.load(eval_path)
        timestep = data['timesteps'][-1]

        last_results = data['results'][-1]
        last_disc_results = data['discounted_results'][-1]
        last_ep_lengths = data['ep_lengths'][-1]

        result = np.mean(last_results)
        disc_result = np.mean(last_disc_results)
        ep_length = np.mean(last_ep_lengths)
        
        # Include resumed sessions, but do not sum concurrent environments.
        t = get_monitor_elapsed_time(os.path.join(FOLDER, run_folder))
        
        # Load name of algo from `hyperparameters.txt`.
        hyper_path = os.path.join(FOLDER, run_folder, 'hyperparameters.txt')
        algo = 'UNKNOWN'
        if os.path.exists(hyper_path):
            with open(hyper_path, 'r') as f:
                for line in f.readlines():
                    if 'ALGO' in line:
                        algo = line.split(':')[-1].strip()
        
        if t is None:
            print(f'RUN_ID: {run_id}, ALGO: {algo}, model evaluated but timing data incomplete')
        else:
            # Check if this was a Lagrangian run
            is_lagrangian = False
            if os.path.exists(hyper_path):
                with open(hyper_path, 'r') as f:
                    for line in f.readlines():
                        if 'USE_LAGRANGIAN' in line and 'True' in line:
                            is_lagrangian = True

            tag = '[Lagrangian]' if is_lagrangian else '[Standard] '
            print(f'{tag} RUN_ID: {run_id}, ALGO: {algo}, '
                f'last timestep={timestep}, '
                f'last return={result:.1f}, '
                f'last disc_return={disc_result:.1f}, '
                f'last ep_length={ep_length}, '
                f't={t/3600:.1f} hours')
        
        successful_runs += 1
    
    except Exception as e:
        print(f'RUN_ID: {run_id}, Error reading results: {e}')
        incomplete_runs += 1

print(f'\nSummary: {successful_runs} runs with results, {incomplete_runs} incomplete/missing')

if __name__ == '__main__':
    print('Done.')
