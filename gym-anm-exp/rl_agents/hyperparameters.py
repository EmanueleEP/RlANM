"""
This file contains the hyperparameter values used for training and
testing RL agents.
"""
import os
import sys
import env

# Environment detection: Colab - local
def is_colab():
    try:
        from google.colab import drive
        return True
    except ImportError:
        return False

# Set BASE_DIR based on environment
if is_colab():
    BASE_DIR = '/content/drive/MyDrive/'
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive montato con successo!")
    except Exception as e:
        print(f"Errore nel montaggio di Google Drive: {e}")
else:
    BASE_DIR = './results/'
    print("local run")


# Simulation environment selection
ENV_ID = 'gym_anm:ANM6Easy-v0'
#ENV_ID = "BCP-v0"
#ENV_ID = "Custom_ANM6Easy-v0"

LOG_DIR = BASE_DIR + ENV_ID + '/'
os.makedirs(LOG_DIR, exist_ok=True)

# Algorithm & training settings
GAMMA = 0.995
POLICY = 'MlpPolicy'
TRAIN_STEPS = 10000
MAX_TRAINING_EP_LENGTH = 5000
EVAL_FREQ = 2000
N_EVAL_EPISODES = 5
MAX_EVAL_EP_LENGTH = 3000
N_ENVS = 1 # number of parallel training environments
TB_LOG_NAME = 'run'

# Constraint handling
USE_LAGRANGIAN = True  # If True, use adaptive Lagrangian multipliers
INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS = True # include thermal overload in costs

# Tolerances and limits (in per‑unit, on top of the base limits already in the simulator)
V_TOLERANCE = 1.0 # voltage violation tolerance (pu)
PQ_LIMIT = 0.5 # active power / branch loading tolerance (pu)
Q_LIMIT = 1.0   

# Lagrangian parameters
LAMBDA_LR = 0.01 # learning rate for dual variables
PEAK_WEIGHT = 0.2 # weight given to peak cost in lambda update (0 = average only)
LAMBDA_MAX = 10.0 # maximum allowed lambda value

# Fixed penalty weights for StandardPenaltyWrapper
# Used when USE_LAGRANGIAN=False and/or --bcp-reward is active.
# These multiply info["constraint_costs"] = [V_violation, P_violation, Q_violation].
STANDARD_PENALTY_W_V = 1.0
STANDARD_PENALTY_W_P = 1.0
STANDARD_PENALTY_W_Q = 1.0

# PPO-specific hyperparameters
PPO_LR_INITIAL = 3e-4
PPO_LR_FINAL   = 3e-5
PPO_CLIP_INITIAL = 0.2
PPO_CLIP_FINAL   = 0.1
PPO_TARGET_KL    = 0.03

# Normalization
NORM_OBS = True
NORM_REWARD = True
CLIP_OBS = 10.0
CLIP_REWARD = 50.0

# Auxiliary wrappers
FORECAST_HORIZON_STEPS = 0 # number of future steps to append (0 = disabled)
PFE_TERMINATE_PENALTY = 50.0 # base penalty for PFE non‑convergence
PFE_LAMBDA_STEPS = 200.0 # extra penalty scaling when Lagrangian is active

# BCP‑reward mode (energy‑only base reward with fixed weights)
USE_BCP_REWARD = False # overridable via CLI --bcp-reward

# SAC-specific hyperparameters (if needed)
SAC_LR = 3e-4

if __name__ == '__main__':
    print('Done.')
