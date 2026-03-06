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
import os
import numpy as np

import gymnasium as gym

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

import torch
from .hyperparameters import *
from .callbacks import ProgressBarManager, EvalCallback
from .utils import save_hyperparameters, parse_args, make_log_dirs  # REMOVED make_envs, added save_hyperparameters
from .wrappers import LagrangianWrapper, TimeLimitWrapper, NormalizeActionWrapper
from .lambda_sync_callback import LambdaSyncCallback

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

# Check if we're continuing training or starting fresh
AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')
ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
EVAL_FILE = os.path.join(LOG_DIR, 'evaluations.npz')
LAMBDA_PATH = os.path.join(LOG_DIR, 'lambda_state.npy')  # MOVED HERE - defined before use

continuing_training = os.path.exists(AGENT_PATH)

# lagrangian parameters
V_TOLERANCE = 0.05
LAMBDA_LR = 1e-3

# Restore lambda from disk if continuing training
lambda_init_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)

if USE_LAGRANGIAN and continuing_training and os.path.exists(LAMBDA_PATH):
    lambda_init_value = np.load(LAMBDA_PATH).astype(np.float32)  # keep shape (3,)
    print(f"Restored lambda from checkpoint: {lambda_init_value}")

if continuing_training:
    print(f"Found existing model at {AGENT_PATH}")
    print("Continuing training from saved checkpoint...")
else:
    print("No existing model found. Starting fresh training...")
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



def make_train_env(rank: int = 0):
    def _init():
        env = gym.make(ENV_ID)
        env = TimeLimitWrapper(env, max_episode_steps=MAX_TRAINING_EP_LENGTH)
        env = NormalizeActionWrapper(env)
        if USE_LAGRANGIAN:
            env = LagrangianWrapper(env, v_tolerance=V_TOLERANCE, lambda_lr=LAMBDA_LR, lambda_init=lambda_init_value)
        env = Monitor(env, LOG_DIR)
        env.reset(seed=SEED + rank)
        return env
    return _init


def make_eval_env(rank: int = 0):
    """Factory that returns a single wrapped eval env (lambda not updated)."""
    def _init():
        env = gym.make(ENV_ID)
        env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
        env = NormalizeActionWrapper(env) 
        # lambda_lr=0 disables dual updates during evaluation
        if USE_LAGRANGIAN:
            env = LagrangianWrapper(env, v_tolerance=V_TOLERANCE, lambda_lr=0.0, lambda_init=lambda_init_value)
        env = Monitor(env, LOG_DIR)
        env.reset(seed=SEED + 1000 + rank)
        return env
    return _init

print("Creating environments...")
try:
    # Use make_vec_env with factory functions - this is the CORRECT way
    train_env = DummyVecEnv([make_train_env(rank=i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env(rank=0)])
    print("Environments created successfully")

except Exception as e:
    print(f"Error creating environments: {e}")
    import traceback
    traceback.print_exc()
    raise

# Load existing evaluation data if continuing
old_data = None
if continuing_training and os.path.exists(EVAL_FILE):
    print("Loading existing evaluation data...")
    try:
        old_data = np.load(EVAL_FILE)
        old_timesteps = list(old_data['timesteps'])
        old_results = list(old_data['results'])
        old_ep_lengths = list(old_data['ep_lengths'])
        old_discounted = list(old_data['discounted_results']) if 'discounted_results' in old_data else None
        print(f"Loaded {len(old_timesteps)} previous evaluation checkpoints")
        print(f"Last evaluation at timestep: {old_timesteps[-1]}")
    except Exception as e:
        print(f"Warning: Could not load evaluation data: {e}")
        old_data = None

# Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=N_EVAL_EPISODES, verbose=True)

# Restore previous evaluation data
if old_data is not None:
    eval_callback.evaluations_timesteps = old_timesteps
    eval_callback.evaluations_results = old_results
    eval_callback.evaluations_length = old_ep_lengths
    if old_discounted is not None:
        eval_callback.evaluation_discounted_results = old_discounted
        if hasattr(eval_callback, '_is_discounted_set'):
            eval_callback._is_discounted_set = True


if USE_LAGRANGIAN and N_ENVS > 1:
    sync_callback = LambdaSyncCallback(train_env, verbose=1)
    callbacks = [eval_callback, sync_callback]
else:
    callbacks = [eval_callback]

print("Creating model...")

# Agent
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    if continuing_training:
        # Load existing model
        model = ALGO.load(AGENT_PATH, env=train_env, device=device)
        print("Loaded existing model successfully")
    else:
        # Create new model
        model = ALGO(POLICY, train_env, gamma=GAMMA, verbose=1, 
                    tensorboard_log=TENSORBOARD_LOG, device=device)
        print("Model created successfully")
except Exception as e:
    print(f"Error creating/loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

if __name__ == "__main__":
    # Train agent
    print(f"{'Continuing' if continuing_training else 'Starting'} training for {TRAIN_STEPS} steps...")
    try:
        with ProgressBarManager(TRAIN_STEPS) as c:
            callbacks += [c]
            model.learn(total_timesteps=TRAIN_STEPS, 
                       tb_log_name=TB_LOG_NAME, 
                       callback=callbacks,
                       reset_num_timesteps=not continuing_training)
        print("Training completed successfully!")

        # Save lambda state if Lagrangian was used
        if USE_LAGRANGIAN:
            try:
                lambdas = []
                while True:
                    if isinstance(inner, LagrangianWrapper):
                        lambdas.append(inner.get_lambda())
                        break
                    if not hasattr(inner, 'env'):
                        break
                    inner = inner.env

                if lambdas:
                    avg_lambda = np.mean(np.stack(lambdas, axis=0), axis=0)
                    np.save(LAMBDA_PATH, avg_lambda)
                    print(f"Saved lambdas={avg_lambda} to {LAMBDA_PATH}")
            except Exception as e:
                print(f"Warning: could not save lambda state: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    print('Done!')