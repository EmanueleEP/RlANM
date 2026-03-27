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
    LOG_DIR as _BASE_LOG_DIR,       
    TRAIN_STEPS as _DEFAULT_TRAIN_STEPS,  
)

from .callbacks import ProgressBarManager, EvalCallback
from .utils import save_hyperparameters, parse_args, make_log_dirs
from .wrappers import (
    LagrangianWrapper,
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
    """Scheduler lineare per learning rate e clip range."""
    initial_value = float(initial_value)
    final_value = float(final_value)

    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def _make_train_env_factory(
    env_id, run_log_dir, seed, max_ep_length,
    use_lagrangian, lambda_init, lambda_lr, v_tolerance, pq_limit, q_limit,
    n_envs, include_branch_overload, forecast_horizon, pfe_penalty, pfe_lambda_steps, peak_weight,
):
    """
    Restituisce una factory per gli environment di training.

    Tutti i parametri sono passati esplicitamente: nessuna variabile globale.
    """
    def make_env(rank: int):
        def _init():
            env = gym.make(env_id)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(include_branch_overload)
            # When Lagrangian is active, remove Q/branch costs from base reward
            # to avoid double-penalization (Lagrangian handles them via dual variables)
            env.unwrapped.use_lagrangian_reward = bool(use_lagrangian)
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
            # Standard path: no extra penalty wrapper needed.
            # The original ANM6Easy reward from super().step() already
            # includes -(e_loss + penalty*lamb) with lamb=100.
            env = PFETerminateWrapper(env, penalty=pfe_penalty, lambda_steps=pfe_lambda_steps)
            env = Monitor(env, run_log_dir)
            env.reset(seed=seed + rank)
            return env
        return _init
    return make_env


def _make_eval_env_factory(
    env_id, run_log_dir, seed, max_ep_length,
    include_branch_overload, forecast_horizon, pfe_penalty,
):
    def make_env(rank: int):
        def _init():
            env = gym.make(env_id)
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

    if args.agent == "PPO":
        ALGO = PPO
    elif args.agent == "SAC":
        ALGO = SAC
    else:
        raise ValueError(f"Agente non supportato: {args.agent}")

    SEED = args.seed

    train_steps: int = int(args.steps) if args.steps is not None else _DEFAULT_TRAIN_STEPS
    if args.steps is not None:
        print(f"TRAIN_STEPS CLI: {train_steps}")

    use_lagrangian: bool = args.lagrangian if args.lagrangian is not None else USE_LAGRANGIAN
    if args.lagrangian is not None:
        print(f"USE_LAGRANGIAN CLI: {use_lagrangian}")

    run_log_dir, tensorboard_log = make_log_dirs(_BASE_LOG_DIR)
    print(f"Log directory: {run_log_dir}")

    V_TOLERANCE     = 1.0
    PQ_LIMIT        = 0.5
    # Q magnitude threshold in normalized space (q_pu / Q_TOL_MAGNITUDE).
    # Physical target: 0.1 pu = 10 MVAr total.
    # Normalized: 0.1 / Q_TOL_MAGNITUDE(0.1) = 1.0.
    # This puts Q on the same scale as V/P costs (both ~0-15 range).
    Q_LIMIT         = 1.0
    LAMBDA_LR       = 0.01
    PEAK_WEIGHT = 0.2

    FORECAST_HORIZON_STEPS = 0
    PFE_TERMINATE_PENALTY  = 50.0   # base penalty (always applied)
    PFE_LAMBDA_STEPS       = 200.0  # dynamic scaling: penalty += sum(lambda*limit) * this

    PPO_LR_INITIAL  = 3e-4
    PPO_LR_FINAL    = 3e-5
    PPO_CLIP_INITIAL = 0.2
    PPO_CLIP_FINAL   = 0.1
    PPO_TARGET_KL    = 0.03

    NORM_OBS     = True
    NORM_REWARD  = True
    CLIP_OBS     = 10.0
    CLIP_REWARD  = 50.0

    # Paths del checkpoint
    agent_path  = os.path.join(run_log_dir, "best_model.zip")
    env_path    = os.path.join(run_log_dir, "training_vec_env")
    eval_file   = os.path.join(run_log_dir, "evaluations.npz")
    lambda_path = os.path.join(run_log_dir, "lambda_state.npy")

    continuing_training = os.path.exists(agent_path)

    lambda_init_value = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    if use_lagrangian and continuing_training and os.path.exists(lambda_path):
        try:
            loaded = np.load(lambda_path).astype(np.float32)
            if loaded.shape == (3,):
                lambda_init_value = np.clip(loaded, 0.0, 500.0)
                print(f"Lambda ripristinata da checkpoint: {lambda_init_value}")
            else:
                print(f"WARNING: lambda shape errata {loaded.shape}, uso defaults.")
        except Exception as e:
            print(f"WARNING: impossibile caricare lambda: {e}. Uso defaults.")

    if not continuing_training:
        save_hyperparameters(run_log_dir, {
            "SEED": SEED, "ENV_ID": ENV_ID, "LOG_DIR": run_log_dir,
            "TRAIN_STEPS": train_steps,
            "MAX_TRAINING_EP_LENGTH": MAX_TRAINING_EP_LENGTH,
            "EVAL_FREQ": EVAL_FREQ, "N_EVAL_EPISODES": N_EVAL_EPISODES,
            "TENSORBOARD_LOG": tensorboard_log, "TB_LOG_NAME": TB_LOG_NAME,
            "POLICY": POLICY, "ALGO": ALGO.__name__,
            "GAMMA": GAMMA, "MAX_EVAL_EP_LENGTH": MAX_EVAL_EP_LENGTH,
            "USE_LAGRANGIAN": use_lagrangian,
            "PEAK_WEIGHT": PEAK_WEIGHT,
            "LAMBDA_LR": LAMBDA_LR,
            "Q_LIMIT": Q_LIMIT,
            "STANDARD_PENALTY_W_V": STANDARD_PENALTY_W_V,
            "STANDARD_PENALTY_W_P": STANDARD_PENALTY_W_P,
            "STANDARD_PENALTY_W_Q": STANDARD_PENALTY_W_Q,
        })

    set_random_seed(SEED)

    train_factory = _make_train_env_factory(
        env_id=ENV_ID, run_log_dir=run_log_dir, seed=SEED,
        max_ep_length=MAX_TRAINING_EP_LENGTH,
        use_lagrangian=use_lagrangian,
        lambda_init=lambda_init_value,
        lambda_lr=LAMBDA_LR,
        v_tolerance=V_TOLERANCE,
        pq_limit=PQ_LIMIT,
        q_limit=Q_LIMIT,
        n_envs=N_ENVS,
        include_branch_overload=INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
        forecast_horizon=FORECAST_HORIZON_STEPS,
        pfe_penalty=PFE_TERMINATE_PENALTY,
        pfe_lambda_steps=PFE_LAMBDA_STEPS if use_lagrangian else 0.0,
        peak_weight=PEAK_WEIGHT,
    )
    eval_factory = _make_eval_env_factory(
        env_id=ENV_ID, run_log_dir=run_log_dir, seed=SEED,
        max_ep_length=MAX_EVAL_EP_LENGTH,
        include_branch_overload=INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS,
        forecast_horizon=FORECAST_HORIZON_STEPS,
        pfe_penalty=PFE_TERMINATE_PENALTY,
    )

    print("Creazione environment...")
    train_env = DummyVecEnv([train_factory(rank=i) for i in range(N_ENVS)])
    eval_env  = DummyVecEnv([eval_factory(rank=0)])

    train_env = VecNormalize(
        train_env, gamma=GAMMA, training=True,
        norm_obs=NORM_OBS, norm_reward=NORM_REWARD,
        clip_obs=CLIP_OBS, clip_reward=CLIP_REWARD,
    )
    eval_env = VecNormalize(
        eval_env, gamma=GAMMA, training=False,
        norm_obs=NORM_OBS, norm_reward=False, clip_obs=CLIP_OBS,
    )
    print("Environment creati.")

    old_data = None
    if continuing_training and os.path.exists(eval_file):
        try:
            old_data = np.load(eval_file)
            print(f"Caricati {len(old_data['timesteps'])} checkpoint precedenti.")
        except Exception as e:
            print(f"WARNING: impossibile caricare evaluations.npz: {e}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=run_log_dir,
        log_path=run_log_dir, eval_freq=EVAL_FREQ,
        deterministic=True, render=False,
        n_eval_episodes=N_EVAL_EPISODES, verbose=True,
    )
    if old_data is not None:
        eval_callback.evaluations_timesteps = list(old_data["timesteps"])
        eval_callback.evaluations_results   = list(old_data["results"])
        eval_callback.evaluations_length    = list(old_data["ep_lengths"])
        if "discounted_results" in old_data:
            eval_callback.evaluation_discounted_results = list(old_data["discounted_results"])

    callbacks = [eval_callback]
    if use_lagrangian and N_ENVS > 1:
        callbacks.append(LambdaSyncCallback(train_env, dual_lr=LAMBDA_LR, verbose=1, peak_weight=PEAK_WEIGHT))
    callbacks.append(BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1))

    if not args.disable_step_log:
        step_log_path = os.path.join(run_log_dir, "step_log.jsonl")
        callbacks.append(StepLoggerCallback(
            step_log_path, log_observations=True,
            log_every=int(args.step_log_every),
        ))
        print(f"Step log: {os.path.abspath(step_log_path)}")

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if continuing_training:
        model = ALGO.load(agent_path, env=train_env, device=device)
        print("Modello caricato da checkpoint.")
    else:
        if args.agent == "PPO":
            model = PPO(
                POLICY, train_env, gamma=GAMMA, verbose=1,
                tensorboard_log=tensorboard_log, device=device,
                learning_rate=_linear_schedule(PPO_LR_INITIAL, PPO_LR_FINAL),
                clip_range=_linear_schedule(PPO_CLIP_INITIAL, PPO_CLIP_FINAL),
                target_kl=PPO_TARGET_KL,
                ent_coef=0.01,
            )
        else:  # SAC
            SAC_LR = 3e-4
            model = SAC(
                POLICY, train_env, gamma=GAMMA, verbose=1,
                tensorboard_log=tensorboard_log, device=device,
                learning_rate=SAC_LR,
                ent_coef="auto",
            )

    # Training
    print(f"{'Riprendendo' if continuing_training else 'Avviando'} training per {train_steps} step...")
    with ProgressBarManager(train_steps) as pbar_callback:
        callbacks.append(pbar_callback)
        model.learn(
            total_timesteps=train_steps,
            tb_log_name=TB_LOG_NAME,
            callback=callbacks,
            reset_num_timesteps=not continuing_training,
        )
    print("Training completato.")

    # Save lambda
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
                assert np.all(avg_lambda >= 0) and np.all(avg_lambda <= 500)
                if os.path.exists(lambda_path):
                    np.save(lambda_path.replace(".npy", "_backup.npy"), np.load(lambda_path))
                np.save(lambda_path, avg_lambda)
                print(f"Lambda salvata: {avg_lambda}")
        except Exception as e:
            print(f"WARNING: impossibile salvare lambda: {e}")


if __name__ == "__main__":
    main()
    print("Done.")