"""
Main training script for RL agents on the BCP/ANM6Easy environment.

It supports:
    - PPO and SAC algorithms,
    - Lagrangian dual‑variable adaptation,
    - Standard penalty baseline,
    - BCP‑style energy‑only reward mode,
    - Step‑by‑step logging,
    - Checkpoint resumption.

All hyperparameters are imported from hyperparameters.py.
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
    LOG_DIR as _BASE_LOG_DIR,
    TRAIN_STEPS as _DEFAULT_TRAIN_STEPS,
    V_TOLERANCE,
    PQ_LIMIT,
    Q_LIMIT,
    LAMBDA_LR,
    PEAK_WEIGHT,
    STANDARD_PENALTY_W_V,
    STANDARD_PENALTY_W_P,
    STANDARD_PENALTY_W_Q,
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
    USE_BCP_REWARD as _USE_BCP_REWARD,
)

from .callbacks import ProgressBarManager, EvalCallback
from .utils import save_hyperparameters, parse_args, make_log_dirs, make_env_with_profile_seed
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


def _make_train_env_factory(
    env_id,
    run_log_dir,
    seed,
    max_ep_length,
    use_lagrangian,
    use_bcp_reward,
    lambda_init,
    lambda_lr,
    v_tolerance,
    pq_limit,
    q_limit,
    w_v,
    w_p,
    w_q,
    n_envs,
    include_branch_overload,
    forecast_horizon,
    pfe_penalty,
    pfe_lambda_steps,
    peak_weight,
):
    """
    Factory that creates a training environment for each parallel rank.
    The same wrapper stack is applied as during training.
    """
    def make_env(rank: int):
        def _init():
            env = gym.make(env_id)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(include_branch_overload)
            # BCP energy‑only base reward: used by both Lagrangian and BCP‑standard paths.
            env.unwrapped.use_lagrangian_reward = bool(use_lagrangian or use_bcp_reward)

            env = TimeLimitWrapper(env, max_episode_steps=max_ep_length)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)

            if forecast_horizon and forecast_horizon > 0:
                env = ForecastObsWrapper(env, horizon_steps=forecast_horizon)

            if use_lagrangian:
                effective_lr = 0.0 if (n_envs > 1) else lambda_lr
                env = LagrangianWrapper(
                    env,
                    v_tolerance=v_tolerance,
                    pq_limit=pq_limit,
                    q_limit=q_limit,
                    lambda_lr=effective_lr,
                    lambda_init=lambda_init,
                    n_envs=n_envs,
                    peak_weight=peak_weight,
                )
            elif use_bcp_reward:
                # BCP energy‑only base + fixed penalty weights (no adaptive dual variables)
                env = StandardPenaltyWrapper(env, w_v=w_v, w_p=w_p, w_q=w_q)
            # ANM6Easy path: no extra wrapper → original reward unchanged.

            env = PFETerminateWrapper(
                env,
                penalty=pfe_penalty,
                lambda_steps=pfe_lambda_steps if use_lagrangian else 0.0,
            )
            env = Monitor(env, run_log_dir)
            env.reset(seed=seed + rank)
            return env
        return _init
    return make_env


def _make_eval_env_factory(
    env_id,
    run_log_dir,
    seed,
    max_ep_length,
    include_branch_overload,
    forecast_horizon,
    pfe_penalty,
    eval_profile_seed: int = 456,
):
    """Factory for evaluation environments – uses a different profile seed."""
    def make_env(rank: int):
        def _init():
            # Use the helper to create the environment
            env = make_env_with_profile_seed(env_id, profile_seed=eval_profile_seed)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(include_branch_overload)
            env = TimeLimitWrapper(env, max_episode_steps=max_ep_length)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env, clip_penalty_weight=0.0)
            if forecast_horizon and forecast_horizon > 0:
                env = ForecastObsWrapper(env, horizon_steps=forecast_horizon)
            env = PFETerminateWrapper(env, penalty=pfe_penalty)
            env = Monitor(env, run_log_dir)
            env.reset(seed=seed + 1000 + rank)
            return env
        return _init
    return make_env


def main():
    args = parse_args()

    # Algorithm selection 
    if args.agent == "PPO":
        ALGO = PPO
    elif args.agent == "SAC":
        ALGO = SAC
    else:
        raise ValueError(f"Agent not supported: {args.agent}")

    SEED = args.seed

    # Override steps from CLI 
    train_steps = int(args.steps) if args.steps is not None else _DEFAULT_TRAIN_STEPS
    if args.steps is not None:
        print(f"TRAIN_STEPS overridden to: {train_steps}")

    # Override Lagrangian flag from CLI 
    use_lagrangian = args.lagrangian if args.lagrangian is not None else USE_LAGRANGIAN
    if args.lagrangian is not None:
        print(f"USE_LAGRANGIAN overridden to: {use_lagrangian}")

    # BCP‑reward mode (energy‑only base reward) 
    use_bcp_reward = args.bcp_reward if args.bcp_reward is not None else _USE_BCP_REWARD
    if use_lagrangian:
        use_bcp_reward = False  # Lagrangian mode takes precedence
    if use_bcp_reward:
        print("BCP‑REWARD mode: energy‑only base + fixed penalty weights (no adaptive lambdas)")

    # Create log directories 
    run_log_dir, tensorboard_log = make_log_dirs(_BASE_LOG_DIR)
    print(f"Log directory: {run_log_dir}")

    # Load previous lambda if resuming 
    agent_path  = os.path.join(run_log_dir, "best_model.zip")
    env_path    = os.path.join(run_log_dir, "training_vec_env")
    eval_file   = os.path.join(run_log_dir, "evaluations.npz")
    lambda_path = os.path.join(run_log_dir, "lambda_state.npy")
    continuing_training = os.path.exists(agent_path)

    lambda_init_value = np.array([0.5, 3.5, 0.5], dtype=np.float32)
    if use_lagrangian and continuing_training and os.path.exists(lambda_path):
        try:
            loaded = np.load(lambda_path).astype(np.float32)
            if loaded.shape == (3,):
                lambda_init_value = np.clip(loaded, 0.0, LAMBDA_MAX)
                print(f"Lambda restored from checkpoint: {lambda_init_value}")
            else:
                print(f"WARNING: lambda shape {loaded.shape} not expected, using defaults.")
        except Exception as e:
            print(f"WARNING: Could not load lambda: {e}. Using defaults.")

    # Save hyperparameters for this run (only if starting fresh) 
    if not continuing_training:
        save_hyperparameters(run_log_dir, {
            "SEED": SEED,
            "ENV_ID": ENV_ID,
            "LOG_DIR": run_log_dir,
            "TRAIN_STEPS": train_steps,
            "MAX_TRAINING_EP_LENGTH": MAX_TRAINING_EP_LENGTH,
            "EVAL_FREQ": EVAL_FREQ,
            "N_EVAL_EPISODES": N_EVAL_EPISODES,
            "TENSORBOARD_LOG": tensorboard_log,
            "TB_LOG_NAME": TB_LOG_NAME,
            "POLICY": POLICY,
            "ALGO": ALGO.__name__,
            "GAMMA": GAMMA,
            "MAX_EVAL_EP_LENGTH": MAX_EVAL_EP_LENGTH,
            "USE_LAGRANGIAN": use_lagrangian,
            "USE_BCP_REWARD": use_bcp_reward,
            "PEAK_WEIGHT": PEAK_WEIGHT,
            "LAMBDA_LR": LAMBDA_LR,
            "Q_LIMIT": Q_LIMIT,
            "STANDARD_PENALTY_W_V": STANDARD_PENALTY_W_V,
            "STANDARD_PENALTY_W_P": STANDARD_PENALTY_W_P,
            "STANDARD_PENALTY_W_Q": STANDARD_PENALTY_W_Q,
        })

    # Set random seed 
    set_random_seed(SEED)

    # Build environment factories 
    train_factory = _make_train_env_factory(
        env_id=ENV_ID,
        run_log_dir=run_log_dir,
        seed=SEED,
        max_ep_length=MAX_TRAINING_EP_LENGTH,
        use_lagrangian=use_lagrangian,
        use_bcp_reward=use_bcp_reward,
        lambda_init=lambda_init_value,
        lambda_lr=LAMBDA_LR,
        v_tolerance=V_TOLERANCE,
        pq_limit=PQ_LIMIT,
        q_limit=Q_LIMIT,
        w_v=STANDARD_PENALTY_W_V,
        w_p=STANDARD_PENALTY_W_P,
        w_q=STANDARD_PENALTY_W_Q,
        n_envs=N_ENVS,
        include_branch_overload=INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
        forecast_horizon=FORECAST_HORIZON_STEPS,
        pfe_penalty=PFE_TERMINATE_PENALTY,
        pfe_lambda_steps=PFE_LAMBDA_STEPS if use_lagrangian else 0.0,
        peak_weight=PEAK_WEIGHT,
    )
    eval_factory = _make_eval_env_factory(
        env_id=ENV_ID,
        run_log_dir=run_log_dir,
        seed=SEED,
        max_ep_length=MAX_EVAL_EP_LENGTH,
        include_branch_overload=INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
        forecast_horizon=FORECAST_HORIZON_STEPS,
        pfe_penalty=PFE_TERMINATE_PENALTY,
    )

    print("Creating environments...")
    train_env = DummyVecEnv([train_factory(rank=i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([eval_factory(rank=0)])

    # Normalization wrappers
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
    print("Environments created.")

    # Load previous evaluation data if resuming
    old_data = None
    if continuing_training and os.path.exists(eval_file):
        try:
            old_data = np.load(eval_file)
            print(f"Loaded {len(old_data['timesteps'])} previous evaluation checkpoints.")
        except Exception as e:
            print(f"WARNING: Could not load evaluations.npz: {e}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_log_dir,
        log_path=run_log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=True,
    )
    if old_data is not None:
        eval_callback.evaluations_timesteps = list(old_data["timesteps"])
        eval_callback.evaluations_results = list(old_data["results"])
        eval_callback.evaluations_length = list(old_data["ep_lengths"])
        if "discounted_results" in old_data:
            eval_callback.evaluation_discounted_results = list(old_data["discounted_results"])

    callbacks = [eval_callback]

    # Lambda synchronization for multi‑env Lagrangian training
    if use_lagrangian and N_ENVS > 1:
        callbacks.append(
            LambdaSyncCallback(
                train_env,
                dual_lr=LAMBDA_LR,
                verbose=1,
                peak_weight=PEAK_WEIGHT,
            )
        )

    # Battery exploration monitor
    callbacks.append(BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1))

    # Step‑by‑step JSONL logger (unless disabled)
    if not args.disable_step_log:
        step_log_path = os.path.join(run_log_dir, "step_log.jsonl")
        callbacks.append(
            StepLoggerCallback(
                step_log_path,
                log_observations=True,
                log_every=int(args.step_log_every),
            )
        )
        print(f"Step log will be written to: {os.path.abspath(step_log_path)}")

    # Model instantiation / loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if continuing_training:
        model = ALGO.load(agent_path, env=train_env, device=device)
        print("Model loaded from checkpoint.")
    else:
        if args.agent == "PPO":
            model = PPO(
                POLICY,
                train_env,
                gamma=GAMMA,
                verbose=1,
                tensorboard_log=tensorboard_log,
                device=device,
                learning_rate=_linear_schedule(PPO_LR_INITIAL, PPO_LR_FINAL),
                clip_range=_linear_schedule(PPO_CLIP_INITIAL, PPO_CLIP_FINAL),
                target_kl=PPO_TARGET_KL,
                ent_coef=0.01,
            )
        else:  # SAC
            model = SAC(
                POLICY,
                train_env,
                gamma=GAMMA,
                verbose=1,
                tensorboard_log=tensorboard_log,
                device=device,
                learning_rate=SAC_LR,
                ent_coef="auto",
            )

    # Training loop
    print(f"{'Resuming' if continuing_training else 'Starting'} training for {train_steps} steps...")
    with ProgressBarManager(train_steps) as pbar_callback:
        callbacks.append(pbar_callback)
        model.learn(
            total_timesteps=train_steps,
            tb_log_name=TB_LOG_NAME,
            callback=callbacks,
            reset_num_timesteps=not continuing_training,
        )
    print("Training completed.")

    # Save final lambda (if Lagrangian)
    if use_lagrangian:
        try:
            inner = train_env
            while hasattr(inner, "venv"):
                inner = inner.venv
            lambdas = []
            for env in getattr(inner, "envs", []):
                w = env
                while hasattr(w, "env"):
                    if isinstance(w, LagrangianWrapper):
                        lambdas.append(w.get_lambda())
                        break
                    w = w.env
            if lambdas:
                avg_lambda = np.mean(np.stack(lambdas, axis=0), axis=0)
                assert avg_lambda.shape == (3,)
                assert np.all(avg_lambda >= 0) and np.all(avg_lambda <= LAMBDA_MAX)
                # Backup old lambda if it exists
                if os.path.exists(lambda_path):
                    np.save(lambda_path.replace(".npy", "_backup.npy"), np.load(lambda_path))
                np.save(lambda_path, avg_lambda)
                print(f"Final lambda saved: {avg_lambda}")
        except Exception as e:
            print(f"WARNING: Could not save lambda: {e}")


if __name__ == "__main__":
    main()
    print("Done.")
