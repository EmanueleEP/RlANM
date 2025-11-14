"""
This script can be used to train PPO or SAC RL agents using
the Stable Baselines3 implementations.

It accepts 2 command line arguments: ::

    $ python -m rl_agents.train <ALGO> -s <SEED>

where
- <ALGO> is the RL algorithm to use in {SAC, PPO},
- <SEED> is a random seed.

All other hyperparameter values are taken from `hyperparameters.py`.
"""
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

import torch
from .hyperparameters import *
from .callbacks import ProgressBarManager, EvalCallback
from .utils import make_envs, save_hyperparameters, parse_args, make_log_dirs


# Read command line arguments.
args = parse_args()
SEED = args.seed

if args.agent == 'PPO':
    ALGO = PPO
elif args.agent == 'SAC':
    ALGO = SAC
else:
    raise ValueError('Unimplemented agent ' + args.agent)

print(f"Training {args.agent} with seed {SEED}")

# Make log directories.
LOG_DIR, TENSORBOARD_LOG = make_log_dirs(LOG_DIR)
print(f"Log directory: {LOG_DIR}")

# Write parameters to .txt file in log directory.
PARAMS = {
    'SEED': SEED,
    'ENV_ID': ENV_ID,
    'LOG_DIR': LOG_DIR,
    'MAX_TRAINING_EP_LENGTH': MAX_TRAINING_EP_LENGTH,
    'EVAL_FREQ': EVAL_FREQ,
    'N_EVAL_EPISODES': N_EVAL_EPISODES,
    'TENSORBOARD_LOG': TENSORBOARD_LOG,
    'TB_LOG_NAME': TB_LOG_NAME,
    'POLICY': POLICY,
    'ALGO': ALGO.__name__,
    'TRAIN_STEPS': TRAIN_STEPS,
    'GAMMA': GAMMA,
    'MAX_EVAL_EP_LENGTH': MAX_EVAL_EP_LENGTH,
}
save_hyperparameters(LOG_DIR, PARAMS)

# Set random seed
set_random_seed(SEED)

print("Creating environments...")
# Environments
try:
    train_env, eval_env = make_envs(ENV_ID, LOG_DIR, GAMMA, MAX_TRAINING_EP_LENGTH, MAX_EVAL_EP_LENGTH, SEED, N_ENVS)
    print("Environments created successfully")
except Exception as e:
    print(f"Error creating environments: {e}")
    import traceback
    traceback.print_exc()
    raise

# Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=N_EVAL_EPISODES, verbose=True)
callbacks = [eval_callback]

print("Creating model...")
# Agent
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model_kwargs = {}
    if ALGO == PPO:
        model_kwargs["n_steps"] = N_STEPS
        model_kwargs["batch_size"] = BATCH_SIZE
    elif ALGO == SAC:
        model_kwargs["train_freq"] = TRAIN_FREQ
        model_kwargs["batch_size"] = BATCH_SIZE

    print(f"Using model kwargs: {model_kwargs}")

    model = ALGO(POLICY, train_env, gamma=GAMMA, verbose=1, 
                 tensorboard_log=TENSORBOARD_LOG, device=device,
                 **model_kwargs) # <-- Pass the new kwargs here
    
    # --- END OF MODIFIED BLOCK ---

    print("Model created successfully")
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
    raise

if __name__ == "__main__":
    # Train agent
    print(f"Starting training for {TRAIN_STEPS} steps...")
    try:
        with ProgressBarManager(TRAIN_STEPS) as c:
            callbacks += [c]
            model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=TB_LOG_NAME, callback=callbacks)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    print('Done!')