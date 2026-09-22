"""
Continue training of a previously saved agent.

This script resumes from a checkpoint created by train.py, reusing the same
environment wrappers, normalization statistics, and (if Lagrangian) lambda values.

Usage:
    python -m rl_agents.continue_training <ALGO> -p <LOG_DIR> -s <SEED> [--steps N]

Saved hyperparameters take precedence over current defaults; explicit CLI options
take precedence over the saved configuration.
"""

import os
from types import SimpleNamespace
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO


from .callbacks import ProgressBarManager, EvalCallback
from .utils import (
    parse_args,
    make_env_with_profile_seed,
    resolve_resume_config,
    make_monitor_session,
    configure_env,
)
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


def main(args=None):

    args = parse_args() if args is None else args
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
    final_agent_path = os.path.join(LOG_DIR, 'final_model.zip')
    final_env_path = os.path.join(LOG_DIR, 'final_vec_env')
    if os.path.exists(final_agent_path) and os.path.exists(final_env_path):
        AGENT_PATH = final_agent_path
        ENV_PATH = final_env_path
        LAMBDA_PATH = os.path.join(LOG_DIR, 'lambda_state.npy')
    else:
        AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')
        ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
        LAMBDA_PATH = os.path.join(LOG_DIR, 'best_lambda_state.npy')
    TENSORBOARD_LOG = os.path.join(LOG_DIR, 'tensorboard')

    if not os.path.exists(AGENT_PATH):
        raise FileNotFoundError(f"No model found at {AGENT_PATH}")

    cfg = SimpleNamespace(**resolve_resume_config(args))
    SEED = cfg.SEED
    print(f"Resume configuration: ENV_ID={cfg.ENV_ID}, "
          f"USE_LAGRANGIAN={cfg.USE_LAGRANGIAN}, USE_BCP_REWARD={cfg.USE_BCP_REWARD}")

    print(f"Continuing training from {AGENT_PATH}")

    if cfg.USE_LAGRANGIAN and not os.path.exists(LAMBDA_PATH):
        print(
            f"WARNING: No lambda checkpoint paired with {AGENT_PATH}; "
            "using the default lambda values."
        )

    # Restore lambda from disk
    lambda_init_value = np.array(
        [cfg.LAMBDA_INIT_V, cfg.LAMBDA_INIT_P, cfg.LAMBDA_INIT_Q], dtype=np.float32
    )
    if cfg.USE_LAGRANGIAN and os.path.exists(LAMBDA_PATH):
        try:
            loaded = np.load(LAMBDA_PATH).astype(np.float32)
            if loaded.shape == (3,):
                lambda_init_value = np.clip(loaded, 0.0, cfg.LAMBDA_MAX)
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
    monitor_dir = make_monitor_session(LOG_DIR)

    # Environment factories (identical to train.py)
    def make_train_env(rank: int = 0):
        def _init():
            env = make_env_with_profile_seed(
                cfg.ENV_ID,
                profile_seed=SEED + rank,
                profile_file=cfg.PROFILE_FILE,
                profile_provider=cfg.PROFILE_PROVIDER,
                profile_split=cfg.PROFILE_SPLIT,
            )
            configure_env(env, vars(cfg), use_lagrangian_reward=cfg.USE_LAGRANGIAN or cfg.USE_BCP_REWARD)

            env = TimeLimitWrapper(env, max_episode_steps=cfg.MAX_TRAINING_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env, clip_penalty_weight=cfg.BATTERY_CLIP_PENALTY)

            if cfg.FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=cfg.FORECAST_HORIZON_STEPS)

            if cfg.USE_LAGRANGIAN:
                effective_lr = 0.0 if (cfg.N_ENVS > 1) else cfg.LAMBDA_LR
                env = LagrangianWrapper(
                    env,
                    v_tolerance=cfg.V_TOLERANCE,
                    pq_limit=cfg.PQ_LIMIT,
                    q_limit=cfg.Q_LIMIT,
                    lambda_lr=effective_lr,
                    lambda_init=lambda_init_value,
                    n_envs=cfg.N_ENVS,
                    peak_weight=cfg.PEAK_WEIGHT,
                    lambda_max=cfg.LAMBDA_MAX,
                )
            elif cfg.USE_BCP_REWARD:
                env = StandardPenaltyWrapper(
                    env,
                    w_v=cfg.STANDARD_PENALTY_W_V,
                    w_p=cfg.STANDARD_PENALTY_W_P,
                    w_q=cfg.STANDARD_PENALTY_W_Q,
                )
            # original/custom: no penalty wrapper, but BatteryClipWrapper and
            # PFETerminateWrapper still add their penalties to the native reward.

            env = PFETerminateWrapper(
                env,
                penalty=cfg.PFE_TERMINATE_PENALTY,
                lambda_steps=cfg.PFE_LAMBDA_STEPS if cfg.USE_LAGRANGIAN else 0.0,
            )
            env = Monitor(env, os.path.join(monitor_dir, f"train_{rank}.monitor.csv"))
            env.reset(seed=SEED + rank)
            return env
        return _init

    def make_eval_env(rank: int = 0):
        """Evaluation env: no penalty wrappers, only the termination penalty."""
        def _init():
            env = make_env_with_profile_seed(
                cfg.ENV_ID,
                profile_seed=456,
                profile_file=cfg.EVAL_PROFILE_FILE,
                profile_provider=cfg.EVAL_PROFILE_PROVIDER,
                profile_split=cfg.EVAL_PROFILE_SPLIT,
            )
            configure_env(env, vars(cfg))
            env = TimeLimitWrapper(env, max_episode_steps=cfg.MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env, clip_penalty_weight=0.0)
            if cfg.FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=cfg.FORECAST_HORIZON_STEPS)
            env = PFETerminateWrapper(env, penalty=cfg.PFE_TERMINATE_PENALTY)
            env = Monitor(env, os.path.join(monitor_dir, f"eval_{rank}.monitor.csv"))
            env.reset(seed=SEED + 1000 + rank)
            return env
        return _init

    print("Creating environments...")
    train_env = DummyVecEnv([make_train_env(rank=i) for i in range(cfg.N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env(rank=0)])

    # Load or create VecNormalize
    if os.path.exists(ENV_PATH):
        print(f"Loading VecNormalize from {ENV_PATH}")
        train_env = VecNormalize.load(ENV_PATH, train_env)
        train_env.training = True
        train_env.norm_reward = cfg.NORM_REWARD
        eval_env = VecNormalize.load(ENV_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("No saved VecNormalize found; creating fresh.")
        train_env = VecNormalize(
            train_env,
            gamma=cfg.GAMMA,
            training=True,
            norm_obs=cfg.NORM_OBS,
            norm_reward=cfg.NORM_REWARD,
            clip_obs=cfg.CLIP_OBS,
            clip_reward=cfg.CLIP_REWARD,
        )
        eval_env = VecNormalize(
            eval_env,
            gamma=cfg.GAMMA,
            training=False,
            norm_obs=cfg.NORM_OBS,
            norm_reward=False,
            clip_obs=cfg.CLIP_OBS,
        )
    print("Environments ready.")

    # Load model (to get current timesteps)
    model = ALGO.load(AGENT_PATH, env=train_env, device=device)
    print("Model loaded.")
    current_timesteps = model.num_timesteps
    print(f"Current model timesteps: {current_timesteps}")

    # Training steps
    additional_steps = cfg.TRAIN_STEPS
    total_timesteps = current_timesteps + additional_steps
    print(f"Additional steps: {additional_steps}, Total after: {total_timesteps}")

    # Override learning rate for continuation
    if args.agent == 'PPO':
        # Compute remaining progress based on total steps, not just the added ones.
        # For simplicity, we use a linear schedule over all total steps.
        # In practice, the schedule is computed based on progress_remaining passed by SB3,
        # which is (total_timesteps - self.num_timesteps) / total_timesteps.
        # Since self.num_timesteps starts at current_timesteps, the progress is computed correctly.
        model.learning_rate = _linear_schedule(cfg.PPO_LR_INITIAL, cfg.PPO_LR_FINAL)
        model.clip_range = _linear_schedule(cfg.PPO_CLIP_INITIAL, cfg.PPO_CLIP_FINAL)
        print(f"PPO LR schedule: {cfg.PPO_LR_INITIAL} → {cfg.PPO_LR_FINAL} over {total_timesteps} total steps")
    else:  # SAC
        model.learning_rate = cfg.SAC_LR
        print(f"SAC LR set to constant: {cfg.SAC_LR}")

    model._setup_lr_schedule()

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=cfg.EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=cfg.N_EVAL_EPISODES,
        verbose=True,
    )

    # Important: align the callback counter with the timesteps already done
    eval_callback.n_calls = current_timesteps
    print(f"EvalCallback.n_calls set to {current_timesteps}")

    # Load old data into the callback
    if old_data is not None:
        eval_callback.restore_evaluations(old_data)

    callbacks = [eval_callback]

    if cfg.USE_LAGRANGIAN and cfg.N_ENVS > 1:
        callbacks.append(
            LambdaSyncCallback(
                train_env,
                dual_lr=cfg.LAMBDA_LR,
                verbose=1,
                peak_weight=cfg.PEAK_WEIGHT,
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
            tb_log_name=cfg.TB_LOG_NAME,
            callback=callbacks,
            reset_num_timesteps=False,
        )

    final_model_path = os.path.join(LOG_DIR, "final_model")
    final_env_path = os.path.join(LOG_DIR, "final_vec_env")
    model.save(final_model_path)
    train_env.save(final_env_path)
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Final VecNormalize state saved to: {final_env_path}")

    # Save final lambda
    if cfg.USE_LAGRANGIAN:
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
                assert np.all(avg_lambda >= 0) and np.all(avg_lambda <= cfg.LAMBDA_MAX)
                if os.path.exists(LAMBDA_PATH):
                    np.save(LAMBDA_PATH.replace('.npy', '_backup.npy'), np.load(LAMBDA_PATH))
                np.save(LAMBDA_PATH, avg_lambda)
                print(f"Saved final lambda: {avg_lambda}")
        except Exception as e:
            print(f"Error saving lambda: {e}")

    print('Done.')


if __name__ == "__main__":
    main()
