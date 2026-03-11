"""
This file contains the hyperparameter values used for training and
testing RL agents.
"""
import os
import sys
import env

def is_colab():
    try:
        from google.colab import drive  # noinspection PyUnresolvedReference
        return True
    except ImportError:
        return False

# Set BASE_DIR based on environment
if is_colab():
    BASE_DIR = '/content/drive/MyDrive/UNI/Tesi/gym-anm-results/'
    # Opzionale: monta Google Drive automaticamente
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
ENV_ID = "BCP-v0"

GAMMA = 0.995
POLICY = 'MlpPolicy'
TRAIN_STEPS = 500000  # FIXED: Was 2000 (only 1-2 episodes), now 500K
MAX_TRAINING_EP_LENGTH = 672  # FIXED: Was 5000 (52 weeks!), now 672 (7 days × 96 steps/day)
EVAL_FREQ = 10000  # Evaluate every 10K steps
N_EVAL_EPISODES = 10  # FIXED: Was 5, now 10 for better statistics
MAX_EVAL_EP_LENGTH = 672  # Match training episode length

# new parameter
N_ENVS = 2  # number of parallel environments

USE_LAGRANGIAN = True  # FIXED: Was False, now True to enforce constraints

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
