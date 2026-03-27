"""
This script can be used to continue training a saved agent previously
trained with `train.py`.

It accepts 3 command line arguments: ::

    $ python -m rl_agents.continue_training <ALGO> -p <PATH> -s <SEED> [--steps N]

where
- <ALGO> is the RL algorithm to use in {SAC, PPO},
- <PATH> is a path to the directory in which the agent was
  saved (same as `LOG_DIR` in `hyperparameters.py`),
- <SEED> is a random seed,
- N is the number of additional training steps (default: TRAIN_STEPS).

All other hyperparameter values are taken from `hyperparameters.py`.
"""
import os
import numpy as np

import gymnasium as gym

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

import torch

from .hyperparameters import (
    ENV_ID, GAMMA, POLICY, EVAL_FREQ, N_EVAL_EPISODES,
    MAX_TRAINING_EP_LENGTH, MAX_EVAL_EP_LENGTH, TB_LOG_NAME,
    USE_LAGRANGIAN, N_ENVS,
    STANDARD_PENALTY_W_V, STANDARD_PENALTY_W_P, STANDARD_PENALTY_W_Q,
    INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
    TRAIN_STEPS as _DEFAULT_TRAIN_STEPS,
)
from .callbacks import ProgressBarManager, EvalCallback
from .utils import parse_args
from .wrappers import (
    LagrangianWrapper,
    StandardPenaltyWrapper,
    TimeLimitWrapper,
    NormalizeActionWrapper,
    BatteryClipWrapper,
    PFETerminateWrapper,
    ForecastObsWrapper,
)
from .lambda_sync_callback import LambdaSyncCallback
from .exploration_callbacks import BatteryExplorationCallback
from .step_logger_callback import StepLoggerCallback


def _linear_schedule(initial_value: float, final_value: float = 0.0):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


if __name__ == '__main__':

    # Read command line arguments.
    args = parse_args()
    SEED = args.seed
    LOG_DIR = args.path

    if LOG_DIR is None:
        raise ValueError("You must specify --path / -p pointing to the run directory.")

    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Paths
    AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')
    ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
    LAMBDA_PATH = os.path.join(LOG_DIR, 'lambda_state.npy')
    TENSORBOARD_LOG = os.path.join(LOG_DIR, 'tensorboard')

    if not os.path.exists(AGENT_PATH):
        raise FileNotFoundError(f"No model found at {AGENT_PATH}")

    print(f"Continuing training from {AGENT_PATH}")

    # ── Parameters (MUST match train.py exactly) ──
    V_TOLERANCE     = 1.0
    PQ_LIMIT        = 0.5
    Q_LIMIT         = 0.01
    LAMBDA_LR       = 0.0001
    PEAK_WEIGHT     = 0.2

    NORM_OBS = True
    NORM_REWARD = True
    CLIP_OBS = 10.0
    CLIP_REWARD = 50.0

    FORECAST_HORIZON_STEPS = 0
    PFE_TERMINATE_PENALTY  = 50.0
    PFE_LAMBDA_STEPS       = 1000.0

    # ── Restore lambda from disk ──
    lambda_init_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if USE_LAGRANGIAN and os.path.exists(LAMBDA_PATH):
        try:
            loaded = np.load(LAMBDA_PATH).astype(np.float32)
            if loaded.shape != (3,):
                print(f"WARNING: Loaded lambda has wrong shape {loaded.shape}. Using defaults.")
            else:
                lambda_init_value = np.clip(loaded, 0.0, 500.0)
                print(f"Restored lambda from checkpoint: {lambda_init_value}")
        except Exception as e:
            print(f"WARNING: Could not load lambda: {e}. Using defaults.")

    # ── Load existing evaluation data ──
    eval_file = os.path.join(LOG_DIR, 'evaluations.npz')
    old_data = None
    old_discounted = None
    if os.path.exists(eval_file):
        print("Loading existing evaluation data...")
        old_data = np.load(eval_file)
        old_timesteps = list(old_data['timesteps'])
        old_results = list(old_data['results'])
        old_ep_lengths = list(old_data['ep_lengths'])
        if 'discounted_results' in old_data:
            old_discounted = list(old_data['discounted_results'])
        print(f"Loaded {len(old_timesteps)} checkpoints, last at timestep {old_timesteps[-1]}")

    set_random_seed(SEED)

    # ── Environment factories (same wrapper stack as train.py) ──

    def make_train_env(rank: int = 0):
        def _init():
            env = gym.make(ENV_ID)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env.unwrapped.use_lagrangian_reward = bool(USE_LAGRANGIAN)
            env = TimeLimitWrapper(env, max_episode_steps=MAX_TRAINING_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)

            if FORECAST_HORIZON_STEPS and FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)

            if USE_LAGRANGIAN:
                effective_lr = 0.0 if (N_ENVS > 1) else LAMBDA_LR
                env = LagrangianWrapper(
                    env,
                    v_tolerance=V_TOLERANCE,
                    pq_limit=PQ_LIMIT,
                    q_limit=Q_LIMIT,
                    lambda_lr=effective_lr,
                    lambda_init=lambda_init_value,
                    n_envs=N_ENVS,
                    peak_weight=PEAK_WEIGHT,
                )
            else:
                env = StandardPenaltyWrapper(
                    env,
                    w_v=STANDARD_PENALTY_W_V,
                    w_p=STANDARD_PENALTY_W_P,
                    w_q=STANDARD_PENALTY_W_Q,
                )
            env = PFETerminateWrapper(
                env,
                penalty=PFE_TERMINATE_PENALTY,
                lambda_steps=PFE_LAMBDA_STEPS if USE_LAGRANGIAN else 0.0,
            )
            env = Monitor(env, LOG_DIR)
            env.reset(seed=SEED + rank)
            return env
        return _init

    def make_eval_env(rank: int = 0):
        """Eval env without penalty wrappers — reports pure economic reward."""
        def _init():
            env = gym.make(ENV_ID)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env, clip_penalty_weight=0.0)
            if FORECAST_HORIZON_STEPS and FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)
            env = PFETerminateWrapper(env, penalty=PFE_TERMINATE_PENALTY)
            env = Monitor(env, LOG_DIR)
            env.reset(seed=SEED + 1000 + rank)
            return env
        return _init

    print("Creating environments...")
    train_env = DummyVecEnv([make_train_env(rank=i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env(rank=0)])

    # Load VecNormalize stats if available, otherwise create fresh
    if os.path.exists(ENV_PATH):
        print(f"Loading VecNormalize stats from {ENV_PATH}")
        train_env = VecNormalize.load(ENV_PATH, train_env)
        train_env.training = True
        train_env.norm_reward = NORM_REWARD
        eval_env = VecNormalize.load(ENV_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("No saved VecNormalize found, creating fresh.")
        train_env = VecNormalize(
            train_env, gamma=GAMMA, training=True,
            norm_obs=NORM_OBS, norm_reward=NORM_REWARD,
            clip_obs=CLIP_OBS, clip_reward=CLIP_REWARD,
        )
        eval_env = VecNormalize(
            eval_env, gamma=GAMMA, training=False,
            norm_obs=NORM_OBS, norm_reward=False, clip_obs=CLIP_OBS,
        )
    print("Environments created successfully")

    # ── Training steps (needed before LR override print) ──
    train_steps = _DEFAULT_TRAIN_STEPS if args.steps is None else int(args.steps)

    # ── Load agent ──
    model = ALGO.load(AGENT_PATH, env=train_env, device=device)
    print("Model loaded successfully")

    # ── Override learning rate for continuation ──
    if args.agent == 'PPO':
        CONTINUE_LR_INITIAL = 3e-4
        CONTINUE_LR_FINAL   = 3e-5
        model.learning_rate = _linear_schedule(CONTINUE_LR_INITIAL, CONTINUE_LR_FINAL)
        print(f"LR schedule reset: {CONTINUE_LR_INITIAL} → {CONTINUE_LR_FINAL} over {train_steps} steps")
    else:  # SAC
        SAC_LR = 3e-4
        model.learning_rate = SAC_LR
        print(f"LR set to constant: {SAC_LR}")

    # ── Callbacks ──
    eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                                 log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                                 deterministic=True, render=False,
                                 n_eval_episodes=N_EVAL_EPISODES, verbose=True)

    if old_data is not None:
        eval_callback.evaluations_timesteps = old_timesteps
        eval_callback.evaluations_results = old_results
        eval_callback.evaluations_length = old_ep_lengths
        if old_discounted is not None:
            eval_callback.evaluation_discounted_results = old_discounted

    callbacks = [eval_callback]
    if USE_LAGRANGIAN and N_ENVS > 1:
        callbacks.append(LambdaSyncCallback(train_env, dual_lr=LAMBDA_LR, verbose=1, peak_weight=PEAK_WEIGHT))
    callbacks.append(BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1))

    if not args.disable_step_log:
        step_log_path = os.path.join(LOG_DIR, "step_log.jsonl")
        callbacks.append(
            StepLoggerCallback(
                step_log_path, log_observations=True,
                log_every=int(args.step_log_every),
            )
        )
        print(f"Step log: {os.path.abspath(step_log_path)}")

    # ── Continue training ──
    print(f"Continuing training for {train_steps} steps...")

    with ProgressBarManager(train_steps) as c:
        callbacks += [c]
        model.learn(total_timesteps=train_steps, tb_log_name=TB_LOG_NAME,
                    callback=callbacks, reset_num_timesteps=False)

    # ── Save lambda state ──
    if USE_LAGRANGIAN:
        try:
            def unwrap_envs(vec_env):
                inner = vec_env
                while hasattr(inner, "venv"):
                    inner = inner.venv
                return getattr(inner, "envs", [])

            lambdas = []
            for env in unwrap_envs(train_env):
                inner = env
                while hasattr(inner, 'env'):
                    if isinstance(inner, LagrangianWrapper):
                        lambdas.append(inner.get_lambda())
                        break
                    inner = inner.env

            if lambdas:
                avg_lambda = np.mean(np.stack(lambdas, axis=0), axis=0)
                assert avg_lambda.shape == (3,)
                assert np.all(avg_lambda >= 0) and np.all(avg_lambda <= 500)
                if os.path.exists(LAMBDA_PATH):
                    np.save(LAMBDA_PATH.replace('.npy', '_backup.npy'), np.load(LAMBDA_PATH))
                np.save(LAMBDA_PATH, avg_lambda)
                print(f"Saved lambdas={avg_lambda} to {LAMBDA_PATH}")
        except Exception as e:
            print(f"Error saving lambda state: {e}")

    print('Done.')
