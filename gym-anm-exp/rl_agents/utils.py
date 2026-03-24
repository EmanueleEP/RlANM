"""
This file contains utility functions used to:
- log hyperparameters to a .txt file,
- parse command line arguments,
- create log directories.
"""
import os
import argparse


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
