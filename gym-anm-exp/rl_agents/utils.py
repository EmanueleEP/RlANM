"""
Utility functions for training, evaluation, and logging.
"""
import os
import argparse
import csv
import glob
import math
import warnings
import gymnasium as gym
import numpy as np


def save_hyperparameters(log_dir, params):
    """Write the list of all hyperparameters to a .txt file in the run directory."""
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        for k in params.keys():
            f.write("'{}':'{}'\n".format(k, params[k]))


def get_training_config():
    """Return serializable training defaults, excluding machine-specific paths."""
    from . import hyperparameters

    return {
        key: value for key, value in vars(hyperparameters).items()
        if key.isupper() and key not in ("BASE_DIR", "LOG_DIR")
        and (value is None or isinstance(value, (str, bool, int, float)))
    }


def load_run_config(run_dir, defaults):
    """Overlay saved scalar hyperparameters on defaults without evaluating code."""
    config = defaults.copy()
    path = os.path.join(run_dir, "hyperparameters.txt")
    if not os.path.isfile(path):
        warnings.warn(f"No saved hyperparameters at {path}; using current defaults.", stacklevel=2)
        return config

    restored = set()
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            key, separator, value = line.strip().partition(":")
            if not separator:
                raise ValueError(f"Invalid hyperparameter line in {path}: {line.strip()!r}")
            key = key.strip().strip("'\"")
            value = value.strip().strip("'\"")
            if key not in defaults:
                continue
            default = defaults[key]
            try:
                if value == "None" and (default is None or key in ("Q_LIMIT", "PPO_TARGET_KL")):
                    parsed = None
                elif isinstance(default, bool):
                    if value.lower() not in ("true", "false"):
                        raise ValueError("expected True or False")
                    parsed = value.lower() == "true"
                elif default is None:
                    parsed = float(value)
                else:
                    parsed = type(default)(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid saved hyperparameter {key}={value!r} in {path}") from error
            config[key] = parsed
            restored.add(key)

    missing = sorted(defaults.keys() - restored)
    if missing:
        warnings.warn(
            "Saved run lacks these parameters; using current defaults: " + ", ".join(missing),
            stacklevel=2,
        )
    return config


def resolve_resume_config(args):
    """Resolve defaults < saved run < explicit CLI options."""
    config = load_run_config(args.path, {**get_training_config(), "SEED": 1, "ALGO": args.agent})
    if config["ALGO"] != args.agent:
        raise ValueError(f"Saved run uses {config['ALGO']}, but {args.agent} was requested.")
    for argument, key in (("seed", "SEED"), ("steps", "TRAIN_STEPS"),
                          ("eval_freq", "EVAL_FREQ"),
                          ("eval_episodes", "N_EVAL_EPISODES"),
                          ("lagrangian", "USE_LAGRANGIAN"),
                          ("bcp_reward", "USE_BCP_REWARD")):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    if config["USE_LAGRANGIAN"]:
        config["USE_BCP_REWARD"] = False
    return config


def make_log_dirs(log_dir):
    """Atomically reserve a numbered run directory inside `log_dir`."""
    os.makedirs(log_dir, exist_ok=True)
    i = 0
    while True:
        run_dir = os.path.join(log_dir, f'run_{i}')
        try:
            os.mkdir(run_dir)
        except FileExistsError:
            i += 1
        else:
            break

    tensorboard_log = os.path.join(run_dir, 'tensorboard')
    os.mkdir(tensorboard_log)

    return run_dir, tensorboard_log


def make_explicit_log_dir(run_dir):
    """Atomically reserve an explicit run directory, refusing to reuse it."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    parent = os.path.dirname(run_dir)
    os.makedirs(parent, exist_ok=True)
    try:
        os.mkdir(run_dir)
    except FileExistsError as error:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Choose a new --run-dir to avoid mixing experiment outputs."
        ) from error

    tensorboard_log = os.path.join(run_dir, 'tensorboard')
    os.mkdir(tensorboard_log)

    return run_dir, tensorboard_log


def make_monitor_session(run_dir):
    """Reserve a new monitor directory for this training invocation."""
    parent = os.path.join(run_dir, "monitors")
    os.makedirs(parent, exist_ok=True)
    session = 0
    while True:
        path = os.path.join(parent, f"session_{session}")
        try:
            os.mkdir(path)
        except FileExistsError:
            session += 1
        else:
            return path


def _read_monitor_elapsed_time(path):
    """Read elapsed time through the last complete, valid monitor record."""
    if not os.path.isfile(path):
        return None
    elapsed = None
    with open(path, newline="") as file:
        if not file.readline().startswith("#"):
            file.seek(0)
        for row in csv.DictReader(file):
            try:
                # Ignore incomplete rows while a running monitor is writing.
                float(row["r"])
                int(row["l"])
                value = float(row["t"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                elapsed = max(elapsed or 0.0, value)
    return elapsed


def get_monitor_elapsed_time(run_dir):
    """Sum recorded session durations without counting parallel envs twice.

    Evaluation logs are excluded. The legacy monitor.csv, if present, is
    retained as the duration recorded before migration to session directories.
    """
    durations = []
    legacy = _read_monitor_elapsed_time(os.path.join(run_dir, "monitor.csv"))
    if legacy is not None:
        durations.append(legacy)
    for session in glob.glob(os.path.join(run_dir, "monitors", "session_*")):
        times = [
            _read_monitor_elapsed_time(path)
            for path in glob.glob(os.path.join(session, "train_*.monitor.csv"))
        ]
        times = [value for value in times if value is not None]
        if times:
            durations.append(max(times))
    return sum(durations) if durations else None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Agent to train")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--path", '-p', type=str, default=None,
        help='The path to the run to continue',
    )
    run_group.add_argument(
        "--run-dir", type=str, default=None,
        help='Exact output directory for a new run; it must not already exist.',
    )
    parser.add_argument("--seed", '-s', type=int, default=None,
                        help='Random seed (default: saved run seed when resuming, otherwise 1)')
    parser.add_argument(
        "--steps",
        type=_positive_int,
        default=None,
        help="Override TRAIN_STEPS for quick runs/debugging (default: use hyperparameters.py).",
    )
    parser.add_argument(
        "--step-log-every",
        type=_positive_int,
        default=1000,
        help="Write one JSONL log row every N timesteps (also logs on episode end).",
    )
    parser.add_argument(
        "--eval-freq",
        type=_positive_int,
        default=None,
        help="Override the number of training timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=_positive_int,
        default=None,
        help="Override the number of episodes per evaluation.",
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
        help="Override USE_LAGRANGIAN from the saved run or hyperparameters.py. "
             "--lagrangian forces Lagrangian, --no-lagrangian forces standard.",
    )
    parser.add_argument(
        "--bcp-reward",
        action="store_true",
        default=None,
        help="Use BCP energy-only base reward with fixed penalty weights instead of "
             "ANM6Easy reward. Only active when --no-lagrangian is set. "
             "When omitted, use USE_BCP_REWARD from the saved run or hyperparameters.py.",
    )

    args = parser.parse_args()
    return args


def _positive_int(value):
    """Argparse type accepting strictly positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
