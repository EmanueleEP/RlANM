"""
This file contains the hyperparameter values used for training and
testing RL agents.
"""
import os
import sys
import env

def is_colab():
    try:
        from google.colab import drive
        return True
    except ImportError:
        return False

# Set BASE_DIR based on environment
if is_colab():
    BASE_DIR = '/content/drive/MyDrive/UNI/Tesi/gym-anm-results/'
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive montato con successo!")
    except Exception as e:
        print(f"Errore nel montaggio di Google Drive: {e}")
else:
    BASE_DIR = './results/'
    print("local run")



#ENV_ID = 'gym_anm:ANM6Easy-v0'
#ENV_ID = 'Ciao-v0'
#ENV_ID = 'SimpleEnv-v0'
#ENV_ID = "BCP-v0"
ENV_ID = "Custom_ANM6Easy-v0"

GAMMA = 0.995
POLICY = 'MlpPolicy'
TRAIN_STEPS = 1500000
MAX_TRAINING_EP_LENGTH = 5000 
EVAL_FREQ = 20000
N_EVAL_EPISODES = 5  
MAX_EVAL_EP_LENGTH = 3000  

# new parameter
N_ENVS = 1 

USE_LAGRANGIAN = True  
# If True, branch thermal overload is included in constraint_costs 
# and therefore enforced by the LagrangianWrapper. If False, it is still logged
# via `constraint/branch_overload_pu_max` but not part of the dual update.

INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS = True

# Fixed penalty weights for StandardPenaltyWrapper (used when USE_LAGRANGIAN=False).
# These multiply info["constraint_costs"] = [V_violation, P_violation, Q_violation].
# For a sensitivity analysis, run training with different weight combinations.
STANDARD_PENALTY_W_V = 1.0   # voltage constraint weight
STANDARD_PENALTY_W_P = 1.0   # active-power / branch-overload constraint weight
STANDARD_PENALTY_W_Q = 1.0   # reactive-power constraint weight

LOG_DIR = BASE_DIR + ENV_ID + '/'
os.makedirs(LOG_DIR, exist_ok=True)

# REMOVE THIS SECTION - let make_log_dirs() handle it
# # Create a new directory for this run.
# i = 0
# while os.path.isdir(LOG_DIR + f'run_{i}/'):
#     i += 1
# LOG_DIR += f'run_{i}/'
# os.makedirs(LOG_DIR, exist_ok=True)

# REMOVE THIS TOO - let make_log_dirs() handle it
# TENSORBOARD_LOG = LOG_DIR + 'tensorboard/'
# os.makedirs(TENSORBOARD_LOG, exist_ok=True)


TB_LOG_NAME = 'run'

if __name__ == '__main__':
    print('Done.')
