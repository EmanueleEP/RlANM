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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

import torch
from .hyperparameters import *
from .callbacks import ProgressBarManager, EvalCallback
from .utils import save_hyperparameters, parse_args, make_log_dirs  # REMOVED make_envs, added save_hyperparameters
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

# No top level arguments parsing here
# Instead we move everything into __main__

def main():
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

    # Allow quick debug runs without editing hyperparameters.py
    if args.steps is not None:
        global TRAIN_STEPS
        TRAIN_STEPS = int(args.steps)
        print(f"Overriding TRAIN_STEPS via CLI: {TRAIN_STEPS}")

    # Make log directories.
    global LOG_DIR, TENSORBOARD_LOG
    LOG_DIR, TENSORBOARD_LOG = make_log_dirs(LOG_DIR)
    print(f"Log directory: {LOG_DIR}")

    # Check if we're continuing training or starting fresh
    AGENT_PATH = os.path.join(LOG_DIR, 'best_model.zip')
    ENV_PATH = os.path.join(LOG_DIR, 'training_vec_env')
    EVAL_FILE = os.path.join(LOG_DIR, 'evaluations.npz')
    LAMBDA_PATH = os.path.join(LOG_DIR, 'lambda_state.npy')

    continuing_training = os.path.exists(AGENT_PATH)

    # lagrangian parameters
    V_TOLERANCE = 0.01
    PQ_LIMIT = 0.00
    LAMBDA_LR = 0.1

    # PPO stability parameters
    PPO_LR_INITIAL = 3e-4
    PPO_LR_FINAL = 3e-5
    PPO_CLIP_INITIAL = 0.2
    PPO_CLIP_FINAL = 0.1
    PPO_TARGET_KL = 0.03

    # VecNormalize parameters (reduces reward/obs scale issues)
    NORM_OBS = True
    NORM_REWARD = True
    CLIP_OBS = 10.0
    CLIP_REWARD = 50.0

    # Lookahead observation features (requires retraining when enabled)
    # Set to 0 to disable (keeps observation space unchanged).
    FORECAST_HORIZON_STEPS = 0

    # Early termination on non-converged power flow
    PFE_TERMINATE_PENALTY = 20.0


    def linear_schedule(initial_value: float, final_value: float = 0.0):
        initial_value = float(initial_value)
        final_value = float(final_value)

        def func(progress_remaining: float) -> float:
            return final_value + progress_remaining * (initial_value - final_value)

        return func

    # Restore lambda from disk if continuing training
    lambda_init_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if USE_LAGRANGIAN and continuing_training and os.path.exists(LAMBDA_PATH):
        try:
            lambda_init_value = np.load(LAMBDA_PATH).astype(np.float32)
            # Validate loaded lambda has correct shape and bounds
            if lambda_init_value.shape != (3,):
                print(f"WARNING: Loaded lambda has wrong shape {lambda_init_value.shape}, expected (3,). Using defaults.")
                lambda_init_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            else:
                # Ensure lambda is within bounds [0, lambda_max]
                lambda_init_value = np.clip(lambda_init_value, 0.0, 500.0)
                print(f"Restored lambda from checkpoint: {lambda_init_value}")
        except Exception as e:
            print(f"WARNING: Could not load lambda checkpoint: {e}. Using defaults.")
            lambda_init_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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
            'USE_LAGRANGIAN': USE_LAGRANGIAN,
            'STANDARD_PENALTY_W_V': STANDARD_PENALTY_W_V,
            'STANDARD_PENALTY_W_P': STANDARD_PENALTY_W_P,
            'STANDARD_PENALTY_W_Q': STANDARD_PENALTY_W_Q,
        }
        save_hyperparameters(LOG_DIR, PARAMS)

    # Set random seed
    set_random_seed(SEED)



    def make_train_env(rank: int):
        def _init():
            env = gym.make(ENV_ID)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env = TimeLimitWrapper(env, max_episode_steps=MAX_TRAINING_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)  # Battery clipping wrapper AFTER normalization

            if FORECAST_HORIZON_STEPS and FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)
            
            env = Monitor(env, LOG_DIR)

            if USE_LAGRANGIAN:
                # When using multiple envs with sync callback, disable individual updates (lambda_lr=0)
                # The callback will handle synchronization and averaging
                # For single env, use LAMBDA_LR to enable gradient ascent
                effective_lambda_lr = 0.0 if (N_ENVS > 1) else LAMBDA_LR
                env = LagrangianWrapper(
                    env,
                    v_tolerance=V_TOLERANCE,
                    pq_limit=PQ_LIMIT,
                    lambda_lr=effective_lambda_lr,
                    lambda_init=lambda_init_value,
                )
            else:
                # Fixed-weight penalty covering the same 3 constraint channels
                env = StandardPenaltyWrapper(
                    env,
                    w_v=STANDARD_PENALTY_W_V,
                    w_p=STANDARD_PENALTY_W_P,
                    w_q=STANDARD_PENALTY_W_Q,
                )
            env = PFETerminateWrapper(env, penalty=PFE_TERMINATE_PENALTY)
            env.reset(seed=SEED + rank)
            return env
        return _init


    def make_eval_env(rank: int):
        """Factory that returns a single wrapped eval env.

        NO penalty wrapper (Lagrangian / StandardPenalty) is applied here so
        that the evaluation reward is the pure economic reward from BCP.
        Constraint violations are still tracked via info["constraint_costs"].
        """
        def _init():
            env = gym.make(ENV_ID)
            if hasattr(env, "include_branch_overload_in_constraints"):
                env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
            env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)  # Battery clipping wrapper AFTER normalization
            if FORECAST_HORIZON_STEPS and FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)
            # No LagrangianWrapper / StandardPenaltyWrapper — eval reports raw economic reward
            env = PFETerminateWrapper(env, penalty=PFE_TERMINATE_PENALTY)
            env = Monitor(env, LOG_DIR)
            env.reset(seed=SEED + 1000 + rank)
            return env
        return _init

    print("Creating environments...")
    try:
        # Use make_vec_env with factory functions - this is the CORRECT way
        train_env = DummyVecEnv([make_train_env(rank=i) for i in range(N_ENVS)])
        eval_env = DummyVecEnv([make_eval_env(rank=0)])
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
            norm_reward=False,  # report true reward during evaluation
            clip_obs=CLIP_OBS,
        )
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
        sync_callback = LambdaSyncCallback(train_env, dual_lr=LAMBDA_LR, verbose=1)
        battery_callback = BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1)
        callbacks = [eval_callback, sync_callback, battery_callback]
    else:
        battery_callback = BatteryExplorationCallback(verbose=1, battery_reward_scale=0.1)
        callbacks = [eval_callback, battery_callback]

    if not args.disable_step_log:
        step_log_path = os.path.join(LOG_DIR, "step_log.jsonl")
        step_logger = StepLoggerCallback(
            step_log_path,
            log_observations=True,
            log_every=int(args.step_log_every),
        )
        print(f"Step log: {os.path.abspath(step_log_path)} (every {int(args.step_log_every)} steps + episode end)")
        callbacks.append(step_logger)

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
            if args.agent == "PPO":
                model = PPO(
                    POLICY,
                    train_env,
                    gamma=GAMMA,
                    verbose=1,
                    tensorboard_log=TENSORBOARD_LOG,
                    device=device,
                    learning_rate=linear_schedule(PPO_LR_INITIAL, PPO_LR_FINAL),
                    clip_range=linear_schedule(PPO_CLIP_INITIAL, PPO_CLIP_FINAL),
                    target_kl=PPO_TARGET_KL,
                )
            else:
                model = ALGO(
                    POLICY,
                    train_env,
                    gamma=GAMMA,
                    verbose=1,
                    tensorboard_log=TENSORBOARD_LOG,
                    device=device,
                    learning_rate=linear_schedule(PPO_LR_INITIAL, PPO_LR_FINAL),
                )
        print("Model created successfully")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Train agent
    print(f"{'Continuing' if continuing_training else 'Starting'} training for {TRAIN_STEPS} steps...")
    try:
        with ProgressBarManager(TRAIN_STEPS) as pbar_callback:
            callbacks.append(pbar_callback)
            model.learn(total_timesteps=TRAIN_STEPS, 
                       tb_log_name=TB_LOG_NAME, 
                       callback=callbacks,
                       reset_num_timesteps=not continuing_training)
        print("Training completed successfully!")

        # Save lambda state if Lagrangian was used
        if USE_LAGRANGIAN:
            try:
                def unwrap_envs(vec_env):
                    inner = vec_env
                    while hasattr(inner, "venv"):
                        inner = inner.venv
                    return getattr(inner, "envs", [])

                lambdas = []
                # Initialize with first environment and traverse wrappers
                base_envs = unwrap_envs(train_env)
                inner = base_envs[0] if base_envs else None
                while hasattr(inner, 'env'):
                    if isinstance(inner, LagrangianWrapper):
                        lambdas.append(inner.get_lambda())
                        break
                    inner = inner.env
                
                # Also check other environments for safety
                for env in base_envs[1:]:
                    inner = env
                    while hasattr(inner, 'env'):
                        if isinstance(inner, LagrangianWrapper):
                            lambdas.append(inner.get_lambda())
                            break
                        inner = inner.env

                if lambdas:
                    avg_lambda = np.mean(np.stack(lambdas, axis=0), axis=0)
                    # Validate before saving
                    assert avg_lambda.shape == (3,), f"Lambda has invalid shape: {avg_lambda.shape}"
                    assert np.all(avg_lambda >= 0.0) and np.all(avg_lambda <= 500.0), f"Lambda out of bounds: {avg_lambda}"
                    
                    # Save with backup
                    lambda_backup_path = LAMBDA_PATH.replace('.npy', '_backup.npy')
                    if os.path.exists(LAMBDA_PATH):
                        np.save(lambda_backup_path, np.load(LAMBDA_PATH))
                    
                    np.save(LAMBDA_PATH, avg_lambda)
                    print(f"Saved lambdas={avg_lambda} to {LAMBDA_PATH}")
                else:
                    print("Warning: No LagrangianWrapper found in training environments")
            except AssertionError as e:
                print(f"Error: Lambda validation failed: {e}. Not saving.")
            except Exception as e:
                print(f"Error: Could not save lambda state: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    print('Done!')
