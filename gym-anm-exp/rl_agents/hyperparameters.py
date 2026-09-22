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
ENV_ID = "gym_anm:ANM6Easy-v0"
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
USE_LAGRANGIAN = False  # The original ANM6Easy environment does not expose Lagrangian costs
INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS = True # include thermal overload in costs

# Tolerances and limits (in per‑unit, on top of the base limits already in the simulator)
V_TOLERANCE = 1.0 # voltage violation tolerance (pu)
PQ_LIMIT = 0.5 # active power / branch loading tolerance (pu)
Q_LIMIT = 0.1 # with Lagrangian/BCP reward: max mean fraction of the reactive capability in use (0.1 = 10%)

# Constraint cost normalization (pu), applied by the environments before the
# costs reach the wrappers. The tolerances above are multiples of these, e.g.
# the effective voltage limit is V_TOLERANCE * V_TOL pu.
V_TOL = 0.01 # voltage violation scale
P_TOL = 0.05 # active power / branch overload scale
Q_TOL = 0.01 # reactive power violation scale (the Lagrangian/BCP Q cost is a capability fraction instead)

# Custom environment reward. The energy terms (W_CURTAIL, W_SLACK_*, W_SOC_BOUNDARY)
# are used by Custom_ANM6Easy-v0 and, with the Lagrangian/BCP reward, by BCP-v0.
# W_Q_MAGNITUDE and W_BRANCH_SOFT apply only to Custom_ANM6Easy-v0 with --method custom
# (BCP-v0 otherwise keeps the original gym-anm reward).
W_CURTAIL = 1.0 # per MWh curtailed
W_SLACK_IMPORT = 1.0 # per MWh imported from the grid
W_SLACK_EXPORT = 0.1 # per MWh exported to the grid
W_SOC_BOUNDARY = 30.0 # quadratic penalty outside [SOC_LOW, SOC_HIGH]
W_Q_MAGNITUDE = 0.5 # per MVArh of reactive power
W_BRANCH_SOFT = 1.0 # per MVAh above BRANCH_SOFT_LOADING
BRANCH_SOFT_LOADING = 0.8 # fraction of the branch rating where the soft penalty starts
SOC_LOW = 0.15 # SoC fraction bounds for Custom_ANM6Easy-v0
SOC_HIGH = 0.95
BCP_SOC_LOW = 0.2 # SoC fraction bounds for BCP-v0
BCP_SOC_HIGH = 0.8

# Lagrangian parameters
LAMBDA_LR = 0.01 # learning rate for dual variables
PEAK_WEIGHT = 0.2 # weight given to peak cost in lambda update (0 = average only)
LAMBDA_MAX = 10.0 # maximum allowed lambda value
LAMBDA_INIT_V = 0.5 # initial lambda for new runs / resumes without a saved lambda
LAMBDA_INIT_P = 3.5
LAMBDA_INIT_Q = 0.5

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
PPO_ENT_COEF     = 0.01

# Normalization
NORM_OBS = True
NORM_REWARD = True
CLIP_OBS = 10.0
CLIP_REWARD = 50.0

# Auxiliary wrappers
BATTERY_CLIP_PENALTY = 10.0 # training penalty per unit of clipped battery action (0 in evaluation)
FORECAST_HORIZON_STEPS = 0 # number of future steps to append (0 = disabled)
PFE_TERMINATE_PENALTY = 50.0 # base penalty for PFE non‑convergence
PFE_LAMBDA_STEPS = 200.0 # extra penalty scaling when Lagrangian is active

# BCP‑reward mode (energy‑only base reward with fixed weights)
USE_BCP_REWARD = False # overridable via CLI --bcp-reward

# SAC-specific hyperparameters (if needed)
SAC_LR = 3e-4

if __name__ == '__main__':
    print('Done.')
