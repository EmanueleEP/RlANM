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
import argparse
import fcntl
import numpy as np
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from rl_agents.hyperparameters import ENV_ID, MAX_EVAL_EP_LENGTH  # ADDED MAX_EVAL_EP_LENGTH
from rl_agents.utils import load_visualization_env
from rl_agents.wrappers import LagrangianWrapper, TimeLimitWrapper, NormalizeActionWrapper, BatteryClipWrapper

import tty, termios

from gym_anm.simulator.components import Generator, StorageUnit

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
        tty.setraw(fd)
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
    # Handle VecEnv
    if hasattr(env, 'envs'):
        base_env = env.envs[0]
    else:
        base_env = env
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
    print("\n")
    print(f"STEP {step}")

    if action_time is not None:
        print(f"ACTION DECISION TIME: {action_time*1000:.2f} ms")

    print("\nACTION TAKEN (Agent Commands):")

    if hasattr(anm_env, 'simulator'):
        sim = anm_env.simulator
        action_idx = 0

        print("  (Each row shows: RAW action - Physical setpoint (MW/MVAr) - Limit status)\n")

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
                    status = "OVERLOAD" if s_mva > branch.rate * simulator.baseMVA else "OK"
                    print(f"  Branch {branch_id}: S={s_mva:.3f} MVA"
                          f" (limit={branch.rate * simulator.baseMVA:.3f} MVA) - status: {status}")

    print(f"\nREWARD: {reward:.4f}")
    print(f"DONE: {done}")


def make_env(env_id, vec_env_path, wrappers=True):
    """
    Create a single (non-vectorized) environment with the same wrappers
    used during training, suitable for visualization.

    Wrapper order MUST match train.py exactly:
        TimeLimitWrapper → NormalizeActionWrapper → BatteryClipWrapper → LagrangianWrapper
    BatteryClipWrapper.step calls self.env.rescale_action, which only exists on
    NormalizeActionWrapper. Placing BatteryClipWrapper before NormalizeActionWrapper
    breaks all battery clipping).
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Load the trained lambda so the agent sees the same reward landscape it
    # was trained on.  Falls back to [1, 1, 1] if the file is missing.
    lambda_init = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lambda_path = os.path.join(vec_env_path, '..', 'lambda_state.npy')
    lambda_path = os.path.normpath(lambda_path)
    if os.path.exists(lambda_path):
        try:
            lambda_init = np.load(lambda_path).astype(np.float32)
            print(f"[make_env] Loaded trained lambda: {lambda_init}")
        except Exception as e:
            print(f"[make_env] Warning: could not load lambda ({e}), using defaults.")
    else:
        print(f"[make_env] lambda_state.npy not found at {lambda_path}, using defaults.")

    def _init():
        env = gym.make(env_id)
        if wrappers:
            # Order must mirror train.py → make_train_env()
            env = TimeLimitWrapper(env, max_episode_steps=MAX_EVAL_EP_LENGTH)
            env = NormalizeActionWrapper(env)
            env = BatteryClipWrapper(env)   # needs NormalizeActionWrapper below it
            # lambda_lr=0.0 keeps the multiplier frozen during visualization
            env = LagrangianWrapper(env, v_tolerance=0.05, lambda_lr=0.0,
                                    lambda_init=lambda_init)
        return env

    return DummyVecEnv([_init])


def visualize(path, algo, T, sleep_time, print_steps, no_mean=False, step_mode=False):
    """Visualize a trained agent."""

    # --- Load model ---
    model = algo.load(os.path.join(path, 'best_model'))

    # --- Build a fresh VecEnv with correct wrappers applied BEFORE vectorization ---
    env = make_env(ENV_ID, os.path.join(path, 'training_vec_env'), wrappers=True)

    # --- Unwrap to the base ANM env for rendering / inspection ---
    anm_env = get_anm_env(env)

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

    print()
    print("IMPORTANT: gym-anm environments render in a BROWSER")
    print("Running on http://127.0.0.1:8050/")
    time.sleep(1)

    # --- Reset: Gymnasium returns (obs, info) ---
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result  # old Gym / SB3 VecEnv returns obs directly

    print("\nInitializing web server...")
    print_env_info(anm_env)
    anm_env.render()
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
                    print_obs(obs[0], anm_env)
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



def print_obs(obs, anm_env):
    """Print a breakdown of the agent's observation vector."""
    obs_flat = obs.flatten()
    sim = anm_env.simulator

    print("\nAGENT OBSERVATION:")
    print(f"  Shape: {obs.shape}")
    if no_mean:
        print(f"  Min: {obs_flat.min():.4f}, Max: {obs_flat.max():.4f}\n")
    else:
        print(f"  Min: {obs_flat.min():.4f}, Max: {obs_flat.max():.4f}, Mean: {obs_flat.mean():.4f}\n")

    print("Bus Voltages (p.u. deviation from 1.0):")
    for i, bus_id in enumerate(sim.buses.keys()):
        print(f"  Bus {bus_id}: {obs_flat[i]:+.4f}")

    idx = sim.N_bus

    print("\nActive Power (Generators & Storage Units, normalized):")
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, Generator) and not dev.is_slack:
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1
        elif isinstance(dev, StorageUnit):
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1

    print("\nReactive Power / SoC (Generators & Storage Units, normalized):")
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, Generator) and not dev.is_slack:
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1
        elif isinstance(dev, StorageUnit):
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1


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