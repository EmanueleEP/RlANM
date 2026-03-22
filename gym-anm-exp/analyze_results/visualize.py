import os
import sys
import time
import warnings
import zipfile
import argparse

import numpy as np
import torch

from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_agents.hyperparameters import (
    ENV_ID, MAX_EVAL_EP_LENGTH, USE_LAGRANGIAN,
    STANDARD_PENALTY_W_V, STANDARD_PENALTY_W_P, STANDARD_PENALTY_W_Q,
)
from rl_agents.wrappers import (
    LagrangianWrapper, StandardPenaltyWrapper,
    TimeLimitWrapper, NormalizeActionWrapper, BatteryClipWrapper,
    PFETerminateWrapper, ForecastObsWrapper,
)


def _validate_policy_architecture(model, env):

    obs_space = env.observation_space
    if not hasattr(obs_space, "shape") or len(obs_space.shape) != 1:
        return

    expected_obs_dim = obs_space.shape[0]

    first_linear = None
    policy = model.policy
    for module in policy.modules():
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break

    if first_linear is None:
        warnings.warn(
            "Impossible to validate the policy architecture: "
            "no linear layer found.",
            stacklevel=2,
        )
        return

    actual_input_dim = first_linear.in_features

    if actual_input_dim != expected_obs_dim:
        raise ValueError(
            f"Mismatch between architecture of the policy and the observation space:\n"
        )


def _load_model_safe(algo_class, zip_path: str, env, device: str):

    try:
        model = algo_class.load(zip_path, env=env, device=device)
        print("[load_model] Load standard SB3: OK")
        _validate_policy_architecture(model, env)
        return model
    except Exception as e:
        warnings.warn(
            f"[load_model] Load standard SB3 failed: {e}\n"
            f"Attempting bypass loading (policy weights only).",
            stacklevel=2,
        )

    model = algo_class("MlpPolicy", env, device=device, verbose=0)

    with zipfile.ZipFile(zip_path, "r") as archive:
        pth_candidates = [n for n in archive.namelist() if n.endswith(".pth")]
        if not pth_candidates:
            raise FileNotFoundError(
                f"No .pth file found in {zip_path}. "
                f"Content: {archive.namelist()}"
            )
        pth_name = pth_candidates[0]
        with archive.open(pth_name) as f:
            tensors = torch.load(f, map_location=model.device, weights_only=False)

    model.policy.load_state_dict(tensors)

    _validate_policy_architecture(model, env)

    print(
        f"[load_model] Bypass loading completed (from '{pth_name}'). "
        f"WARNING: model hyperparameters not verified."
    )
    return model


def make_env(env_id, vec_env_path, wrappers=True):
    try:
        from rl_agents.hyperparameters import FORECAST_HORIZON_STEPS
    except ImportError:
        FORECAST_HORIZON_STEPS = 0

    lambda_init = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lambda_path = os.path.normpath(os.path.join(vec_env_path, "..", "lambda_state.npy"))
    if os.path.exists(lambda_path):
        try:
            lambda_init = np.load(lambda_path).astype(np.float32)
            print(f"[make_env] Lambda critic: {lambda_init}")
        except Exception as e:
            warnings.warn(f"[make_env] Impossibile to load lambdas: {e}. Used default ones.")
    else:
        warnings.warn(f"[make_env] lambda_state.npy not found in {lambda_path}. Used default ones.")

    from rl_agents.hyperparameters import INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS

    def _init():
        import gymnasium as gym
        env = gym.make(env_id)
        if hasattr(env, "include_branch_overload_in_constraints"):
            env.include_branch_overload_in_constraints = bool(INCLUDE_BRANCH_OVERLOAD_IN_CONSTRAINTS)
        if wrappers:
            env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)
            if FORECAST_HORIZON_STEPS and FORECAST_HORIZON_STEPS > 0:
                env = ForecastObsWrapper(env, horizon_steps=FORECAST_HORIZON_STEPS)
            if USE_LAGRANGIAN:
                env = LagrangianWrapper(env, lambda_lr=0.0, lambda_init=lambda_init)
            else:
                env = StandardPenaltyWrapper(
                    env,
                    w_v=STANDARD_PENALTY_W_V,
                    w_p=STANDARD_PENALTY_W_P,
                    w_q=STANDARD_PENALTY_W_Q,
                )
            env = PFETerminateWrapper(env, penalty=20.0)
        return env

    env = DummyVecEnv([_init])

    if vec_env_path and os.path.exists(vec_env_path):
        try:
            env = VecNormalize.load(vec_env_path, env)
            env.training = False
            env.norm_reward = False
            print(f"[make_env] VecNormalize loaded from: {vec_env_path}")
        except Exception as e:
            warnings.warn(
                f"[make_env] Impossibile to load VecNormalize from {vec_env_path}: {e}. "
                "Continuing without normalization."
            )
    else:
        warnings.warn(
            f"[make_env] VecNormalize not found in {vec_env_path}. "
            "Observations might not correspond to those seen during training."
        )

    return env


def visualize(path, algo_class, T, sleep_time, print_steps, no_mean=False, step_mode=False):
    vec_env_path = os.path.join(path, "training_vec_env")
    zip_path     = os.path.join(path, "best_model.zip")
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(ENV_ID, vec_env_path, wrappers=True)

    model = _load_model_safe(algo_class, zip_path, env, device)

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open("policy.pth") as f:
            tensors = torch.load(f, map_location=model.device, weights_only=False)
            
    model.policy.load_state_dict(tensors)
    # --- Unwrap to the base ANM env for rendering / inspection ---
    anm_env = get_anm_env(env)

    # --- Reset: Gymnasium returns (obs, info) ---
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result  # old Gym / SB3 VecEnv returns obs directly

    print("\nInitializing web server...")
    print_env_info(anm_env)
    
    # Render FIRST before replacing sys.stdout to prevent multiprocessing segfaults/pickling errors
    anm_env.render()
    
    # --- Logging setup ---
    log_filename = os.path.join(path, f'visualization_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')
    log_file = open(log_filename, 'w')

    class Output:
        def __init__(self, file, print_to_console=True):
            self.file = file
            self.stdout = sys.stdout
            self.print_to_console = print_to_console

        def write(self, text):
            if self.print_to_console:
                self.stdout.write(text)
            self.file.write(text)

        def flush(self):
            if self.print_to_console:
                self.stdout.flush()
            self.file.flush()

    origin_stdout = sys.stdout
    sys.stdout = Output(log_file, print_to_console=print_steps)

    print("\nStarting agent interaction...")

    done, state = False, None
    action_times = []

    try:
        for i in range(T):
            start_time = time.time()
            action, state = model.predict(obs, state=state, deterministic=True)
            action_time = time.time() - start_time
            action_times.append(action_time)

            obs, rewards, dones, info = env.step(action)

            print_step_info(i + 1, action[0], obs[0], rewards[0], dones[0], info, anm_env, action_time)
            anm_env.render()
            
            if step_mode:
                key = wait_for_keypress(step_mode=True)
                if key is False:
                    break
                if key == 'r':
                    print("\nReset forzato dall'utente.")
                    reset_result = env.reset()
                    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                    state = None
                    continue
                elif key == 'i':
                    print("\n" + "="*80)
                    print_obs(obs[0], anm_env, no_mean, vec_env=env)
                    print("="*80)
                    continue  # Don't step, just show info and wait for next key
            else:
                time.sleep(sleep_time)

            if dones[0]:
                print(f"\nEpisode finished at step {i + 1}")
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
                state = None

    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")

    # Timing statistics
    if action_times:
        mean_t = sum(action_times) / len(action_times)
        std_t = (sum((t - mean_t) ** 2 for t in action_times) / len(action_times)) ** 0.5
        print("\nACTION DECISION TIME STATISTICS")
        print(f"Total steps: {len(action_times)}")
        if not no_mean:
            print(f"Mean time:   {mean_t * 1000:.2f} ms")
        print(f"Min time:    {min(action_times) * 1000:.2f} ms")
        print(f"Max time:    {max(action_times) * 1000:.2f} ms")
        print(f"Std dev:     {std_t * 1000:.2f} ms")

    sys.stdout = origin_stdout
    log_file.close()

    print("The web server will shut down.")
    env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Classe dell'agente da visualizzare")
    parser.add_argument("--path", "-p", type=str, help="Path alla cartella del run")
    parser.add_argument("--sleep", "-s", type=float, default=0.5)
    parser.add_argument("-T", type=int, default=int(1e4))
    parser.add_argument("--print-steps", action="store_true")
    parser.add_argument("--no-mean", action="store_true")
    parser.add_argument("--step-by-step", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ALGO = {"PPO": PPO, "SAC": SAC}.get(args.agent)
    if ALGO is None:
        raise ValueError(f"Agente non supportato: {args.agent}")
    visualize(args.path, ALGO, args.T, args.sleep, args.print_steps, args.no_mean, args.step_by_step)
    print("Done.")