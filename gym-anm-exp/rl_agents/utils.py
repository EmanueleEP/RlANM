"""
Utility functions for training, evaluation, and logging.
"""
import os
import argparse
import gymnasium as gym
import numpy as np


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
    parser.add_argument(
        "--lagrangian",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override USE_LAGRANGIAN from hyperparameters.py. "
             "--lagrangian forces Lagrangian, --no-lagrangian forces standard.",
    )
    parser.add_argument(
        "--bcp-reward",
        action="store_true",
        default=None,
        help="Use BCP energy-only base reward with fixed penalty weights instead of "
             "ANM6Easy reward. Only active when --no-lagrangian is set. "
             "When omitted, use USE_BCP_REWARD from hyperparameters.py.",
    )

    args = parser.parse_args()
    return args


def make_env_with_profile_seed(env_id: str, profile_seed: int = None,
                               solar_scale: float = None, load_scale: float = None,
                               **kwargs):
    """
    Create an environment, optionally passing profile_seed, solar_scale, load_scale.
    Some environments (like BCP-v0) accept them, others (like ANM6Easy) don't.

    The function first tries to create the environment with all provided kwargs.
    If a TypeError occurs, it removes the optional parameters (profile_seed,
    solar_scale, load_scale) and retries. Finally, if the environment has those
    attributes, it sets them manually.
    """
    # Build kwargs dictionary with all optional parameters
    base_kwargs = kwargs.copy()
    if profile_seed is not None:
        base_kwargs['profile_seed'] = profile_seed
    if solar_scale is not None:
        base_kwargs['solar_scale'] = solar_scale
    if load_scale is not None:
        base_kwargs['load_scale'] = load_scale

    # List of parameters that might be unsupported
    optional_params = ['profile_seed', 'solar_scale', 'load_scale']

    # First attempt: try with all parameters
    try:
        return gym.make(env_id, **base_kwargs)
    except TypeError:
        # If it fails, remove the optional parameters and try again
        reduced_kwargs = {k: v for k, v in base_kwargs.items() if k not in optional_params}
        try:
            env = gym.make(env_id, **reduced_kwargs)
        except TypeError:
            # If still fails, try with no extra kwargs at all
            env = gym.make(env_id)

        # Manually set attributes on the unwrapped environment if they exist
        if profile_seed is not None and hasattr(env.unwrapped, 'profile_seed'):
            env.unwrapped.profile_seed = profile_seed
        if solar_scale is not None and hasattr(env.unwrapped, 'solar_scale'):
            env.unwrapped.solar_scale = solar_scale
        if load_scale is not None and hasattr(env.unwrapped, 'load_scale'):
            env.unwrapped.load_scale = load_scale

        return env
