"""
This file contains utility functions used to:
- initialize environments (for training, evaluation, and visualization),
- log hyperparameters to a .txt file,
- parse command line arguments.
"""
import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit 
import argparse
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .wrappers import NormalizeActionWrapper, TimeLimitWrapper

def make_envs(env_id, log_dir, gamma, max_train_ep_length, max_eval_ep_length, seed, n_envs):
    """Return properly vectorized training and evaluation environments."""

    def make_train_env():
        env = gym.make(env_id)
        env.reset(seed=seed)
        
        # USE OFFICIAL GYMNASIUM TimeLimit wrapper instead of your custom one
        env = TimeLimit(env, max_episode_steps=max_train_ep_length) 
        
        env = Monitor(env, log_dir)
        env = NormalizeActionWrapper(env)
        return env

    def make_eval_env():
        env = gym.make(env_id)
        env.reset(seed=seed)
        
        # USE OFFICIAL GYMNASIUM TimeLimit wrapper instead of your custom one
        env = TimeLimit(env, max_episode_steps=max_eval_ep_length)
        
        env = Monitor(env)
        env = NormalizeActionWrapper(env)
        return env

    # Wrap with DummyVecEnv and VecNormalize **only once**
    train_env = DummyVecEnv([make_train_env]* n_envs)
    train_env = VecNormalize(train_env, gamma=gamma)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, gamma=gamma, training=False, norm_reward=False)

    return train_env, eval_env


def load_training_env(env_id, env_path, log_dir, max_train_ep_length, seed, n_envs):
    """Load a saved vectorized training env (used to continue training)."""

    def make_train_env():
        """Helper function to create a single training env."""
        env = gym.make(env_id)
        env.reset(seed=seed)
        
        # USE OFFICIAL GYMNASIUM TimeLimit wrapper
        env = TimeLimit(env, max_episode_steps=max_train_ep_length) 
        
        env = Monitor(env, log_dir)
        env = NormalizeActionWrapper(env)
        return env

    # Re-create the parallel env structure
    env = SubprocVecEnv([make_train_env] * n_envs)
    
    # Load the saved VecNormalize statistics
    env = VecNormalize.load(env_path, env)

    env.reset()
    return env


def load_visualization_env(env_id, env_path, n_envs):
    """Create a visualization env with wrappers (no VecNormalize to load)."""
    from .wrappers import LagrangianWrapper, TimeLimitWrapper
    from rl_agents.hyperparameters import MAX_EVAL_EP_LENGTH
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make():
        #env = gym.make(env_id, render_mode="human")
        env = gym.make(env_id)
        env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
        env = LagrangianWrapper(env, v_tolerance=0.05, lambda_lr=0.0)
        return env

    return DummyVecEnv([_make for _ in range(n_envs)])


def save_hyperparameters(log_dir, params):
    """Write the list of all hyperparameters to a .txt file in the run directory."""
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        for k in params.keys():
            f.write("'{}':'{}'\n".format(k, params[k]))


def make_log_dirs(log_dir):
    """Create new subdirectories in `log_dir` for this run."""
    i = 0
    while os.path.isdir(os.path.join(log_dir, f'run_{i}')):
        i += 1
    log_dir = os.path.join(log_dir, f'run_{i}')
    os.makedirs(log_dir, exist_ok=True)

    # Make tensorboard log directory within `log_dir`.
    tensorboard_log = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_log, exist_ok=True)

    return log_dir, tensorboard_log


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Agent to train")
    parser.add_argument("--path", '-p', type=str, default=None, help='The path to the run to continue')
    parser.add_argument("--seed", '-s', type=int, help='Random seed', default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override TRAIN_STEPS for quick runs/debugging (default: use hyperparameters.py).",
    )
    parser.add_argument(
        "--step-log-every",
        type=int,
        default=1000,
        help="Write one JSONL log row every N timesteps (also logs on episode end).",
    )
    parser.add_argument(
        "--disable-step-log",
        action="store_true",
        help="Disable step-level JSONL logging.",
    )

    args = parser.parse_args()

    return args
