"""
This script can be used to visualize a trained-agent interacting
with an environment, assuming the environment supports rendering.

Usage
-----
python -m analyze_results.visualize <ALGO> -p <PATH> -T <TIMESTEPS>
"""
import os
import sys
import time
import numpy as np
import torch

import zipfile
import argparse
import fcntl
import warnings

from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_agents.hyperparameters import (
    ENV_ID, MAX_EVAL_EP_LENGTH, USE_LAGRANGIAN,
    STANDARD_PENALTY_W_V, STANDARD_PENALTY_W_P, STANDARD_PENALTY_W_Q,
)
from rl_agents.utils import load_visualization_env
from rl_agents.wrappers import (
    LagrangianWrapper, StandardPenaltyWrapper,
    TimeLimitWrapper, NormalizeActionWrapper, BatteryClipWrapper,
    PFETerminateWrapper, ForecastObsWrapper,
)

import faulthandler
faulthandler.enable()

import tty, termios

from gym_anm.simulator.components import Generator, StorageUnit


def _validate_policy_architecture(model, env):

    obs_space = env.observation_space
    if not hasattr(obs_space, "shape") or len(obs_space.shape) != 1:
        # Spazi non-Box o multidimensionali: skip della validazione
        return
 
    expected_obs_dim = obs_space.shape[0]
 
    # Trova il primo layer lineare della policy (mlp_extractor o actor)
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
            f"Mismatch between policy architecture and observation_space:\n"
            f"  Policy input dim:       {actual_input_dim}\n"
            f"  Observation space dim:  {expected_obs_dim}\n"
        )


def _load_model_safe(algo_class, zip_path: str, env, device: str):
    try:
        model = algo_class.load(zip_path, env=env, device=device, 
                                custom_objects={"learning_rate": 3e-4,
                                                "clip_range": 0.2})
        _validate_policy_architecture(model, env)
        return model
    except Exception as e:
        warnings.warn(
            f"[load_model] Load standard SB3 failed: {e}\n"
            f"Tried bypass (just the policy weights).",
            stacklevel=2,
        )
 
    model = algo_class("MlpPolicy", env, device=device, verbose=0)
 
    with zipfile.ZipFile(zip_path, "r") as archive:
        pth_candidates = [n for n in archive.namelist() if n.endswith(".pth")]
        if not pth_candidates:
            raise FileNotFoundError(
                f"No file .pth found in {zip_path}. "
                f"Content: {archive.namelist()}"
            )
        pth_name = pth_candidates[0]
        with archive.open(pth_name) as f:
            tensors = torch.load(f, map_location=model.device, weights_only=False)
 
    model.policy.load_state_dict(tensors)
 
    _validate_policy_architecture(model, env)
 
    print(
        f"[load_model] Bypass loading completed (from '{pth_name}'). "
        f"WARNING: Model hyperparameters not verified."
    )
    return model


def wait_for_keypress(step_mode):
    """
    In step mode: wait for user to press a key.
      ENTER / SPACE → next step
      'q'           → exit visualization
      'r'           → reset episode
    In auto mode: returns True immediately (uses sleep instead).
    Returns True to continue, False to exit.
    """
    if not step_mode:
        return True

    print("\n[STEP MODE] Premi INVIO/SPAZIO=avanti | q=esci | r=reset | i=info", end="", flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        # Clear any buffered input before reading
        import fcntl, os
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        try:
            while sys.stdin.read(1):
                pass
        except:
            pass
        # Reset to blocking mode
        fcntl.fcntl(fd, fcntl.F_SETFL, 0)

        # Now read the actual keypress
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print()

    if ch in ('q', 'Q', '\x03'):  # q, Q, or Ctrl+C
        return False
    # Return the character (Space, Enter, 'r', 'i', etc.)
    return ch  


def get_anm_env(env):
    # Unwrap a (possibly vectorized) env to reach the base ANM environment
    base_env = env
    # Handle VecNormalize / VecEnv wrappers
    while hasattr(base_env, "venv"):
        base_env = base_env.venv
    if hasattr(base_env, "envs"):
        base_env = base_env.envs[0]
    # Unwrap Monitor / Gymnasium wrappers
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env

def print_env_info(anm_env):
    """Print information about the environment's action and observation spaces."""
    print("\n")
    print("ENVIRONMENT INFORMATION")
    print("\n")

    # Action space info
    print("\nACTION SPACE:")

    if not hasattr(anm_env, 'action_space'):
        print("  No action space found.")
        return

    print(f"  Type: {type(anm_env.action_space)}")
    print(f"  Shape: {anm_env.action_space.shape}\n")

    if hasattr(anm_env, 'simulator'):
        sim = anm_env.simulator
        idx = 0
        print("  Each action dimension corresponds to a controllable device:\n")
        for dev_id, dev in sim.devices.items():
            dev_type = type(dev).__name__

            if isinstance(dev, Generator) and not dev.is_slack:
                p_range = (dev.p_min * sim.baseMVA, dev.p_max * sim.baseMVA)
                q_range = (dev.q_min * sim.baseMVA, dev.q_max * sim.baseMVA)
                print(f"  Device {dev_id} ({dev_type}):")
                print(f"    Action[{idx}] → P: range {p_range} MW")
                idx += 1
                print(f"    Action[{idx}] → Q: range {q_range} MVAr")
                idx += 1

            elif isinstance(dev, StorageUnit):
                p_range = (dev.p_min * sim.baseMVA, dev.p_max * sim.baseMVA)
                q_range = (dev.q_min * sim.baseMVA, dev.q_max * sim.baseMVA)
                print(f"  Device {dev_id} ({dev_type}):")
                print(f"    Action[{idx}] → P: range {p_range} MW")
                idx += 1
                print(f"    Action[{idx}] → Q: range {q_range} MVAr")
                idx += 1
    else:
        print(f"  Raw Range: low={anm_env.action_space.low}, high={anm_env.action_space.high}")

    if hasattr(anm_env, 'simulator'):
        simulator = anm_env.simulator

        print("\nCONTROLLABLE DEVICES:")
        for dev_id, dev in simulator.devices.items():
            dev_type = type(dev).__name__
            if dev_type in ['ClassicalGen', 'RenewableGen', 'StorageUnit']:
                if hasattr(dev, 'is_slack') and dev.is_slack:
                    continue
                p_range = f"[{dev.p_min * simulator.baseMVA:.2f}, {dev.p_max * simulator.baseMVA:.2f}] MW"
                q_range = f"[{dev.q_min * simulator.baseMVA:.2f}, {dev.q_max * simulator.baseMVA:.2f}] MVAr"
                print(f"  Device {dev_id} ({dev_type}): P={p_range}, Q={q_range}")

        print("\nBUSES (Voltage Measurement Points):")
        for bus_id, bus in simulator.buses.items():
            bus_type = "Slack" if bus.is_slack else "PQ"
            v_range = f"[{bus.v_min:.3f}, {bus.v_max:.3f}] p.u."
            print(f"  Bus {bus_id} ({bus_type}): V={v_range}, BaseKV={bus.baseKV}")

        print("\nLOADS:")
        for dev_id, dev in simulator.devices.items():
            if type(dev).__name__ == 'Load':
                print(f"  Device {dev_id}: Connected to Bus {dev.bus_id}")
        print("\n")


def print_step_info(step, action, obs, reward, done, info, anm_env, action_time=None):
    """Print detailed information about the current step."""
    info0 = info[0] if isinstance(info, (list, tuple)) and info else (info or {})
    print("\n")
    print(f"STEP {step}")

    if action_time is not None:
        print(f"ACTION DECISION TIME: {action_time*1000:.2f} ms")

    print("\nACTION COMMANDED (Agent Output, before BatteryClipWrapper):")

    if hasattr(anm_env, 'simulator'):
        sim = anm_env.simulator
        action_idx = 0

        print("  (Each row shows: RAW action [-1,1] → Intended setpoint (MW/MVAr) → Limit status)")
        print("  (Actual executed values may differ due to SoC/power clipping — see DEVICE POWER below)\n")

        for dev_id, dev in sim.devices.items():
            is_gen = isinstance(dev, Generator) and not dev.is_slack
            is_des = isinstance(dev, StorageUnit)

            if not (is_gen or is_des):
                continue

            if action_idx < len(action):
                raw_a_p = action[action_idx]
                p_min = dev.p_min * sim.baseMVA
                p_max = dev.p_max * sim.baseMVA
                p_set = p_min + (raw_a_p + 1) / 2 * (p_max - p_min)
                clipped = ""
                if p_set <= p_min + 1e-6: clipped = "(CLIPPED_MIN)"
                if p_set >= p_max - 1e-6: clipped = "(CLIPPED_MAX)"
                print(f"  Device {dev_id} ({type(dev).__name__})  P:"
                      f" raw={raw_a_p:+.3f} → {p_set:.2f} MW  {clipped}")
                action_idx += 1

            if hasattr(dev, "q_min") and hasattr(dev, "q_max"):
                if action_idx < len(action):
                    raw_a_q = action[action_idx]
                    q_min = dev.q_min * sim.baseMVA
                    q_max = dev.q_max * sim.baseMVA
                    q_set = q_min + (raw_a_q + 1) / 2 * (q_max - q_min)
                    clipped = ""
                    if q_set <= q_min + 1e-6: clipped = "(CLIPPED_MIN)"
                    if q_set >= q_max - 1e-6: clipped = "(CLIPPED_MAX)"
                    print(f"                          Q:"
                          f" raw={raw_a_q:+.3f} → {q_set:.2f} MVAr {clipped}")
                    action_idx += 1
    else:
        print(f"  Raw action: {action}")

    print("\nVOLTAGE (PER UNIT) AT BUSES:")
    if hasattr(anm_env, 'simulator'):
        simulator = anm_env.simulator

        if hasattr(simulator, 'state') and simulator.state is not None:
            bus_v_magn = simulator.state.get('bus_v_magn', None)
            if bus_v_magn is not None and 'pu' in bus_v_magn:
                for bus_id, v in bus_v_magn['pu'].items():
                    bus = simulator.buses[bus_id]
                    status = "VIOLATION" if v < bus.v_min or v > bus.v_max else "OK"
                    bus_type = "Slack" if bus.is_slack else "PQ"
                    print(f"  Bus {bus_id} ({bus_type}): {v:.4f} p.u. - status: {status}")

            print("\nDEVICE POWER INJECTIONS:")
            dev_p = simulator.state.get('dev_p', None)
            dev_q = simulator.state.get('dev_q', None)
            if dev_p is not None and 'MW' in dev_p:
                for dev_id, dev in simulator.devices.items():
                    dev_type = type(dev).__name__
                    p_mw = dev_p['MW'].get(dev_id, 0)
                    q_mvar = dev_q['MVAr'].get(dev_id, 0) if dev_q and 'MVAr' in dev_q else 0

                    if dev_type == 'StorageUnit':
                        soc_state = simulator.state.get('des_soc', {})
                        soc = soc_state.get('MWh', {}).get(dev_id, 0) if soc_state else 0
                        print(f"  Device {dev_id} ({dev_type}): P={p_mw:.3f} MW, Q={q_mvar:.3f} MVAr, SoC={soc:.3f} MWh")
                    else:
                        print(f"  Device {dev_id} ({dev_type}): P={p_mw:.3f} MW, Q={q_mvar:.3f} MVAr")

            print("\nBRANCH POWER FLOWS:")
            branch_s = simulator.state.get('branch_s', None)
            if branch_s is not None and 'MVA' in branch_s:
                for branch_id, branch in simulator.branches.items():
                    s_mva = branch_s['MVA'].get(branch_id, 0)
                    status = "OVERLOAD" if abs(s_mva) > branch.rate * simulator.baseMVA else "OK"
                    print(f"  Branch {branch_id}: S={s_mva:.3f} MVA"
                          f" (limit={branch.rate * simulator.baseMVA:.3f} MVA) - status: {status}")

    # Extra diagnostics from info dict (available with or without LagrangianWrapper)
    if isinstance(info0, dict) and info0:
        constraint_costs = info0.get("constraint_costs", None)
        if constraint_costs is not None:
            try:
                c = np.asarray(constraint_costs, dtype=float).flatten()
                if c.size >= 3:
                    print("\nCONSTRAINT COSTS:")
                    print(f"  cost_v={c[0]:.4f}  cost_p={c[1]:.4f}  cost_q={c[2]:.4f}")
            except Exception:
                pass

        br_over = info0.get("constraint/branch_overload_pu_max", None)
        if br_over is not None:
            try:
                print(f"  branch_overload_pu_max={float(br_over):.4f}")
            except Exception:
                pass

        # Explain common "strange" mismatches: availability + battery SoC clipping.
        try:
            sim = anm_env.simulator
            # Battery SoC ceiling effect
            soc_mwh = sim.state.get("des_soc", {}).get("MWh", {}).get(3, None) if sim.state else None
            soc_max_mwh = float(getattr(sim.devices.get(3), "soc_max", 1.0)) * float(sim.baseMVA)
            if soc_mwh is not None and soc_max_mwh > 0:
                if float(soc_mwh) >= soc_max_mwh - 1e-3 and float(action[2]) < -1e-6:
                    print("  note: battery action clipped by SoC ceiling (battery is full).")
            # Renewable availability effect
            gen_p_max = sim.state.get("gen_p_max", {}).get("MW", {}).get(2, None) if sim.state else None
            if gen_p_max is not None:
                print(f"  gen_p_max_MW(device 2)={float(gen_p_max):.3f}")
        except Exception:
            pass

    print(f"\nREWARD: {reward:.4f}")
    print(f"DONE: {done}")

def make_env(env_id, vec_env_path, wrappers=True):
    """Costruisce l'environment di visualizzazione con gli stessi wrapper del training."""
    try:
        from rl_agents.hyperparameters import FORECAST_HORIZON_STEPS
    except ImportError:
        FORECAST_HORIZON_STEPS = 0
 
    lambda_init = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lambda_path = os.path.normpath(os.path.join(vec_env_path, "..", "lambda_state.npy"))
    if os.path.exists(lambda_path):
        try:
            lambda_init = np.load(lambda_path).astype(np.float32)
            print(f"[make_env] Lambda caricata: {lambda_init}")
        except Exception as e:
            warnings.warn(f"[make_env] Impossibile caricare lambda: {e}. Uso defaults.")
    else:
        warnings.warn(f"[make_env] lambda_state.npy non trovato in {lambda_path}. Uso defaults.")
 
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
            print(f"[make_env] VecNormalize caricata da: {vec_env_path}")
        except Exception as e:
            warnings.warn(
                f"[make_env] Impossibile caricare VecNormalize da {vec_env_path}: {e}. "
                "Continuo senza normalizzazione."
            )
    else:
        warnings.warn(
            f"[make_env] VecNormalize non trovata in {vec_env_path}. "
            "Le osservazioni potrebbero non corrispondere a quelle viste durante il training."
        )
 
    return env

def visualize(path, algo_class, T, sleep_time, print_steps, no_mean=False, step_mode=False):
    """Visualizza un agente addestrato."""

    vec_env_path = os.path.join(path, "training_vec_env")
    zip_path     = os.path.join(path, "best_model.zip")
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    env   = make_env(ENV_ID, vec_env_path, wrappers=True)
    model = _load_model_safe(algo_class, zip_path, env, device)

    anm_env = get_anm_env(env)

    # Reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    print("\nInitializing web server...")
    print_env_info(anm_env)

    # Render FIRST before replacing sys.stdout
    anm_env.render()

    # Logging setup
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
                    continue
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

def print_obs(obs, anm_env, no_mean, vec_env=None):
    """Print a breakdown of the agent's observation vector.

    Notes
    -----
    - When VecNormalize is enabled, `obs` is *normalized* (what the policy sees).
      For debugging physical values (MW/MVAr/MWh), we try to also display the
      unnormalized observation via `VecNormalize.unnormalize_obs`.
    - For `gym-anm` environments with `observation="state"`, the observation is
      ordered following `anm_env.obs_values` (dev_p, dev_q, des_soc, gen_p_max, aux, ...).
    """
    obs_flat = obs.flatten()
    sim = anm_env.simulator

    # Try to recover unnormalized obs (physical units) if VecNormalize is active.
    obs_un = None
    if vec_env is not None and hasattr(vec_env, "unnormalize_obs"):
        try:
            obs_un = vec_env.unnormalize_obs(obs.reshape(1, -1))[0].flatten()
        except Exception:
            try:
                obs_un = vec_env.unnormalize_obs(obs).flatten()
            except Exception:
                obs_un = None

    def _print_stats(name, x):
        x = np.asarray(x, dtype=float).flatten()
        print(f"\n{name}:")
        print(f"  Shape: {tuple(x.shape)}")
        if x.size == 0:
            return
        if no_mean:
            print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
        else:
            print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")

    print("\nAGENT OBSERVATION")
    _print_stats("  Policy obs (possibly normalized)", obs_flat)
    if obs_un is not None:
        _print_stats("  Env obs (VecNormalize unnormalized)", obs_un)

    # Decode the observation vector using the env-provided schema (robust to ordering).
    schema = getattr(anm_env, "obs_values", None)
    if not schema:
        print("\n(obs_values unavailable; cannot decode observation vector reliably.)")
        return

    def _decode(vec, title):
        vec = np.asarray(vec, dtype=float).flatten()
        print(f"\n{title}:")
        idx = 0
        for key, ids, unit in schema:
            n = len(ids)
            seg = vec[idx : idx + n]
            idx += n

            unit_str = "" if unit is None else f" {unit}"
            if key in ("dev_p", "dev_q"):
                print(f"  {key}{unit_str}:")
                for dev_id, v in zip(ids, seg):
                    dev = sim.devices.get(dev_id, None)
                    dev_name = type(dev).__name__ if dev is not None else "Unknown"
                    print(f"    Device {dev_id} ({dev_name}): {v:+.4f}{unit_str}")
            elif key == "des_soc":
                print(f"  {key}{unit_str}:")
                for dev_id, v in zip(ids, seg):
                    dev = sim.devices.get(dev_id, None)
                    dev_name = type(dev).__name__ if dev is not None else "Unknown"
                    print(f"    Device {dev_id} ({dev_name}): {v:.4f}{unit_str}")
            elif key == "gen_p_max":
                print(f"  {key}{unit_str}:")
                for dev_id, v in zip(ids, seg):
                    dev = sim.devices.get(dev_id, None)
                    dev_name = type(dev).__name__ if dev is not None else "Unknown"
                    print(f"    Device {dev_id} ({dev_name}): {v:.4f}{unit_str}")
            elif key == "aux":
                print("  aux:")
                for aux_id, v in zip(ids, seg):
                    print(f"    aux[{aux_id}]: {v:.4f}")
            else:
                print(f"  {key}{unit_str}: {seg}")

        if idx != vec.size:
            print(f"  (warning) decoded {idx} elements but obs has {vec.size}.")

    # Prefer decoding the *unnormalized* observation when available.
    if obs_un is not None:
        _decode(obs_un, "Decoded observation (physical units)")
        _decode(obs_flat, "Decoded observation (policy / normalized)")
    else:
        _decode(obs_flat, "Decoded observation")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Class of agent to visualize")
    parser.add_argument("--path", '-p', type=str, help='The path to the folder containing the trained agent')
    parser.add_argument("--sleep", '-s', type=float, default=0.5, help='Sleep time between steps')
    parser.add_argument("-T", type=int, default=int(1e4), help='Number of timesteps to render')
    parser.add_argument("--print-steps", action="store_true", help="Print step info to console (always saved to log)")
    parser.add_argument("--no-mean", action="store_true", help="Hide mean values from output")
    parser.add_argument("--step-by-step", action="store_true", help="Step-by-step simulation")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    visualize(args.path, ALGO, args.T, args.sleep, args.print_steps, args.no_mean, args.step_by_step)
    print('Done.')