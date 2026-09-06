"""
Continue training of a previously saved agent.

This script resumes from a checkpoint created by train.py, reusing the same
environment wrappers, normalization statistics, and (if Lagrangian) lambda values.

Usage:
    python -m rl_agents.continue_training <ALGO> -p <LOG_DIR> -s <SEED> [--steps N]

All hyperparameters are taken from hyperparameters.py.
"""

import os
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from .hyperparameters import (
    ENV_ID,
    GAMMA,
    POLICY,
    EVAL_FREQ,
    N_EVAL_EPISODES,
    MAX_TRAINING_EP_LENGTH,
    MAX_EVAL_EP_LENGTH,
    TB_LOG_NAME,
    USE_LAGRANGIAN,
    N_ENVS,
    STANDARD_PENALTY_W_V,
    STANDARD_PENALTY_W_P,
    STANDARD_PENALTY_W_Q,
    INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
    TRAIN_STEPS as _DEFAULT_TRAIN_STEPS,
    V_TOLERANCE,
    PQ_LIMIT,
    Q_LIMIT,
    LAMBDA_LR,
    PEAK_WEIGHT,
    PPO_LR_INITIAL,
    PPO_LR_FINAL,
    PPO_CLIP_INITIAL,
    PPO_CLIP_FINAL,
    PPO_TARGET_KL,
    SAC_LR,
    NORM_OBS,
    NORM_REWARD,
    CLIP_OBS,
    CLIP_REWARD,
    FORECAST_HORIZON_STEPS,
    PFE_TERMINATE_PENALTY,
    PFE_LAMBDA_STEPS,
    LAMBDA_MAX,
    USE_BCP_REWARD,
)

from .callbacks import ProgressBarManager, EvalCallback
from .utils import parse_args, make_env_with_profile_seed
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
    """Linear schedule for learning rate or clip range."""
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


if __name__ == '__main__':

    args = parse_args()
    SEED = args.seed
    LOG_DIR = args.path

    if LOG_DIR is None:
        raise ValueError("You must specify --path / -p pointing to the run directory.")

    # Algorithm
    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Paths
    AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')
    ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
    LAMBDA_PATH = os.path.join(LOG_DIR, 'lambda_state.npy')
    TENSORBOARD_LOG = os.path.join(LOG_DIR, 'tensorboard')

    if not os.path.exists(AGENT_PATH):
        raise FileNotFoundError(f"No model found at {AGENT_PATH}")

    print(f"Continuing training from {AGENT_PATH}")

    # Restore lambda from disk
    lambda_init_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if USE_LAGRANGIAN and os.path.exists(LAMBDA_PATH):
        try:
            loaded = np.load(LAMBDA_PATH).astype(np.float32)
            if loaded.shape == (3,):
                lambda_init_value = np.clip(loaded, 0.0, LAMBDA_MAX)
                print(f"Restored lambda: {lambda_init_value}")
            else:
                print(f"WARNING: lambda shape {loaded.shape} unexpected, using defaults.")
        except Exception as e:
            print(f"WARNING: Could not load lambda: {e}. Using defaults.")

    # Load existing evaluation data
    eval_file = os.path.join(LOG_DIR, 'evaluations.npz')
    old_data = None
    if os.path.exists(eval_file):
        print("Loading existing evaluation data...")
        with np.load(eval_file) as archive:
            old_data = {key: archive[key] for key in archive.files}
        print(f"Loaded {len(old_data['timesteps'])} evaluation checkpoints.")

    # Set seed
    set_random_seed(SEED)

    # Environment factories (identical to train.py)
    def make_train_env(rank: int = 0):
        def _init():
            env = gym.make(ENV_ID)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env.unwrapped.use_lagrangian_reward = bool(USE_LAGRANGIAN or USE_BCP_REWARD)

            env = TimeLimitWrapper(env, max_episode_steps=MAX_TRAINING_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)

            if FORECAST_HORIZON_STEPS > 0:
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
            elif USE_BCP_REWARD:
                env = StandardPenaltyWrapper(
                    env,
                    w_v=STANDARD_PENALTY_W_V,
                    w_p=STANDARD_PENALTY_W_P,
                    w_q=STANDARD_PENALTY_W_Q,
                )
            # For standard ANM6Easy runs, preserve the environment's native reward.

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
        """Evaluation env: no penalty wrappers, only the termination penalty."""
        def _init():
            env = make_env_with_profile_seed(ENV_ID, profile_seed=456)  # fixed seed for eval
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env, clip_penalty_weight=0.0)
            if FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)
            env = PFETerminateWrapper(env, penalty=PFE_TERMINATE_PENALTY)
            env = Monitor(env, LOG_DIR)
            env.reset(seed=SEED + 1000 + rank)
            return env
        return _init

    print("Creating environments...")
    train_env = DummyVecEnv([make_train_env(rank=i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env(rank=0)])

    # Load or create VecNormalize
    if os.path.exists(ENV_PATH):
        print(f"Loading VecNormalize from {ENV_PATH}")
        train_env = VecNormalize.load(ENV_PATH, train_env)
        train_env.training = True
        train_env.norm_reward = NORM_REWARD
        eval_env = VecNormalize.load(ENV_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("No saved VecNormalize found; creating fresh.")
        train_env = VecNormalize(
            train_env,
            gamma=GAMMA,
            training=True,
            norm_obs=NORM_OBS,
            norm_reward=NORM_REWARD,
            clip_obs=CLIP_OBS,
            clip_reward=CLIP_REWARD,
        )
        eval_env = VecNormalize(
            eval_env,
            gamma=GAMMA,
            training=False,
            norm_obs=NORM_OBS,
            norm_reward=False,
            clip_obs=CLIP_OBS,
        )
    print("Environments ready.")

    # Load model (to get current timesteps)
    model = ALGO.load(AGENT_PATH, env=train_env, device=device)
    print("Model loaded.")
    current_timesteps = model.num_timesteps
    print(f"Current model timesteps: {current_timesteps}")

    # Training steps
    additional_steps = _DEFAULT_TRAIN_STEPS if args.steps is None else int(args.steps)
    total_timesteps = current_timesteps + additional_steps
    print(f"Additional steps: {additional_steps}, Total after: {total_timesteps}")

    # Override learning rate for continuation
    if args.agent == 'PPO':
        # Compute remaining progress based on total steps, not just the added ones.
        # For simplicity, we use a linear schedule over all total steps.
        # In practice, the schedule is computed based on progress_remaining passed by SB3,
        # which is (total_timesteps - self.num_timesteps) / total_timesteps.
        # Since self.num_timesteps starts at current_timesteps, the progress is computed correctly.
        model.learning_rate = _linear_schedule(PPO_LR_INITIAL, PPO_LR_FINAL)
        model.clip_range = _linear_schedule(PPO_CLIP_INITIAL, PPO_CLIP_FINAL)
        print(f"PPO LR schedule: {PPO_LR_INITIAL} → {PPO_LR_FINAL} over {total_timesteps} total steps")
    else:  # SAC
        model.learning_rate = SAC_LR
        print(f"SAC LR set to constant: {SAC_LR}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=True,
    )

    # Important: align the callback counter with the timesteps already done
    eval_callback.n_calls = current_timesteps
    print(f"EvalCallback.n_calls set to {current_timesteps}")

    # Load old data into the callback
    if old_data is not None:
        eval_callback.restore_evaluations(old_data)

    callbacks = [eval_callback]

    if USE_LAGRANGIAN and N_ENVS > 1:
        callbacks.append(
            LambdaSyncCallback(
                train_env,
                dual_lr=LAMBDA_LR,
                verbose=1,
                peak_weight=PEAK_WEIGHT,
            )
        )

    callbacks.append(BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1))

    if not args.disable_step_log:
        step_log_path = os.path.join(LOG_DIR, "step_log.jsonl")
        callbacks.append(
            StepLoggerCallback(
                step_log_path,
                log_observations=True,
                log_every=int(args.step_log_every),
            )
        )
        print(f"Step log: {os.path.abspath(step_log_path)}")

    # Progress bar with correct initial value
    with ProgressBarManager(total_timesteps, initial=current_timesteps) as pbar_callback:
        callbacks.append(pbar_callback)
        model.learn(
            total_timesteps=additional_steps,
            tb_log_name=TB_LOG_NAME,
            callback=callbacks,
            reset_num_timesteps=False,
        )

    # Save final lambda
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
                assert np.all(avg_lambda >= 0) and np.all(avg_lambda <= LAMBDA_MAX)
                if os.path.exists(LAMBDA_PATH):
                    np.save(LAMBDA_PATH.replace('.npy', '_backup.npy'), np.load(LAMBDA_PATH))
                np.save(LAMBDA_PATH, avg_lambda)
                print(f"Saved final lambda: {avg_lambda}")
        except Exception as e:
            print(f"Error saving lambda: {e}")

    print('Done.')
