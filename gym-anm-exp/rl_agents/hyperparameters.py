"""
This file contains the hyperparameter values used for training and
testing RL agents.
"""
import os

# new parameter
N_ENVS = 10  # number of parallel environments

BASE_DIR = './results/'
ENV_ID = 'gym_anm:ANM6Easy-v0'
GAMMA = 0.995
POLICY = 'MlpPolicy'
TRAIN_STEPS = 3000000
MAX_TRAINING_EP_LENGTH = 5000
EVAL_FREQ = 1000
N_EVAL_EPISODES = 5
MAX_EVAL_EP_LENGTH = 3000
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


# --- GPU OPTIMIZATION PARAMETERS ---
# Number of steps to collect from each env before updating the policy (for PPO)
# A larger value means more efficient batching for the GPU.
N_STEPS = 4096 

# Batch size for gradient update. This is THE key parameter for GPU util.
# Default is 64 (PPO) or 256 (SAC), which is tiny.
# Increase this significantly.
BATCH_SIZE = 2048

# How often to update the model (for SAC).
# Default is (1, "step"), which is a performance disaster for parallel envs.
# We change it to (256, "step") to collect 256 steps before one update.
TRAIN_FREQ = (256, "step")


TB_LOG_NAME = 'run'

if __name__ == '__main__':
    print('Done.')