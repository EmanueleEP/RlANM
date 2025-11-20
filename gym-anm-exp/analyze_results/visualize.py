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
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from rl_agents.hyperparameters import ENV_ID
from rl_agents.utils import load_visualization_env

from gym_anm.simulator.components import Generator, StorageUnit


def get_anm_env(env):
    base_env = env.envs[0]
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

            # Only show non-slack generators and storage units (controllable)
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
        # fallback for Box spaces without a simulator
        print(f"  Raw Range: low={anm_env.action_space.low}, high={anm_env.action_space.high}")
    
    # Get device and bus information
    if hasattr(anm_env, 'simulator'):
        simulator = anm_env.simulator
        
        print("\nCONTROLLABLE DEVICES:")
        for dev_id, dev in simulator.devices.items():
            dev_type = type(dev).__name__
            # Only show non-slack generators and storage units (controllable devices)
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


def print_step_info(step, action, obs, reward, done, anm_env):
    """Print detailed information about the current step."""
    print("\n")
    print(f"STEP {step}")
    # Action information
    print("\nACTION TAKEN (Agent Commands):")

    if hasattr(anm_env, 'simulator'):
        sim = anm_env.simulator
        action_idx = 0

        print("  (Each row shows: RAW action → Physical setpoint (MW/MVAr) → Limit status)\n")

        for dev_id, dev in sim.devices.items():
            dev_type = type(dev).__name__

            # Identify controllable devices
            is_gen = isinstance(dev, Generator) and not dev.is_slack
            is_des = isinstance(dev, StorageUnit)

            if not (is_gen or is_des):
                continue

            # Active power control
            if action_idx < len(action):
                raw_a_p = action[action_idx]
                # Map raw action [-1,1] → physical bound
                p_min = dev.p_min * sim.baseMVA
                p_max = dev.p_max * sim.baseMVA
                p_set = p_min + (raw_a_p + 1)/2 * (p_max - p_min)
                clipped = ""
                if p_set <= p_min + 1e-6: clipped = "(CLIPPED_MIN)"
                if p_set >= p_max - 1e-6: clipped = "(CLIPPED_MAX)"

                print(f"  Device {dev_id} ({dev_type})  P:"
                    f" raw={raw_a_p:+.3f} → {p_set:.2f} MW  {clipped}")
                action_idx += 1

            # Reactive power control (if available)
            if hasattr(dev, "q_min") and hasattr(dev, "q_max"):
                if action_idx < len(action):
                    raw_a_q = action[action_idx]
                    q_min = dev.q_min * sim.baseMVA
                    q_max = dev.q_max * sim.baseMVA
                    q_set = q_min + (raw_a_q + 1)/2 * (q_max - q_min)
                    clipped = ""
                    if q_set <= q_min + 1e-6: clipped = "(CLIPPED_MIN)"
                    if q_set >= q_max - 1e-6: clipped = "(CLIPPED_MAX)"

                    print(f"                          Q:"
                        f" raw={raw_a_q:+.3f} → {q_set:.2f} MVAr {clipped}")
                    action_idx += 1

    else:
        print(f"  Raw action: {action}")
    
    # Voltage/State information
    print("\nVOLTAGE (PER UNIT) AT BUSES:")
    if hasattr(anm_env, 'simulator'):
        simulator = anm_env.simulator

        # print("\n")
        # print("Simulator - get_state_space()")

        # print(f"state space is: \n{simulator.get_state_space()}")
        # print("\n")

        
        if hasattr(simulator, 'state') and simulator.state is not None:
            bus_v_magn = simulator.state.get('bus_v_magn', None)
            if bus_v_magn is not None and 'pu' in bus_v_magn:
                for bus_id, v in bus_v_magn['pu'].items():
                    bus = simulator.buses[bus_id]
                    status = "OK"
                    if v < bus.v_min or v > bus.v_max:
                        status = "VIOLATION"
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
                    status = "OK"
                    if s_mva > branch.rate * simulator.baseMVA:
                        status = "OVERLOAD"
                    print(f"  Branch {branch_id}: S={s_mva:.3f} MVA (limit={branch.rate * simulator.baseMVA:.3f} MVA) - status: {status}")

    print(f"\nREWARD: {reward:.4f}")
    print(f"DONE: {done}")

def print_obs(obs, anm_env):
    obs_flat = obs.flatten()
    sim = anm_env.simulator

    # Compute sizes
    n_buses = sim.N_bus
    n_gens = sim.N_non_slack_gen
    n_des = sim.N_des

    print("\nAGENT OBSERVATION:")
    print(f"  Shape: {obs.shape}")
    print(f"  Min: {obs_flat.min():.4f}, Max: {obs_flat.max():.4f}, Mean: {obs_flat.mean():.4f}\n")

    # 1. Bus voltages
    print("Bus Voltages (p.u. deviation from 1.0):")
    for i, bus_id in enumerate(sim.buses.keys()):
        print(f"  Bus {bus_id}: {obs_flat[i]:+.4f}")

    idx = n_buses

    # 2. Active power of generators and storage units
    print("\nActive Power (Generators & Storage Units, normalized):")
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, Generator) and not dev.is_slack:
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1
        elif isinstance(dev, StorageUnit):
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1

    # 3. Reactive power / SoC of generators and storage units
    print("\nReactive Power / SoC (Generators & Storage Units, normalized):")
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, Generator) and not dev.is_slack:
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1
        elif isinstance(dev, StorageUnit):
            print(f"  Device {dev_id} ({type(dev).__name__}): {obs_flat[idx]:+.4f}")
            idx += 1

def visualize(path, algo, T, sleep_time, print_steps):
    """
    Visualize a trained agent.
    """
    # Load agent and environment
    model = algo.load(os.path.join(path, 'best_model'))
    env = load_visualization_env(ENV_ID, os.path.join(path, 'training_vec_env'), 1)

    # Create log file
    log_filename = os.path.join(path, f'visualization_log_{time.strftime("%Y\\%m\\%d\\%H:%M:%S")}.txt')
    log_file = open(log_filename, 'w')

    print()
    print("IMPORTANT: gym-anm environments render in a WEB BROWSER")
    print("'Running on http://127.0.0.1:8050/'")
    print("\nStarting visualization in 3 seconds")
    time.sleep(3)

    anm_env = get_anm_env(env)

    # Console output optional
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
    
    originStdout = sys.stdout
    sys.stdout = Output(log_file, print_to_console=print_steps)

    # Reset and first step
    obs = env.reset()

    print("\nInitializing web server...")
    print_env_info(anm_env)

    anm_env.render()
    
    print("\nStarting agent interaction...")
    
    done, state = False, None

    try:
        for i in range(T):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            print_step_info(i+1, action[0], obs[0], rewards[0], dones[0], anm_env)

            anm_env.render()
                        
            time.sleep(sleep_time)
            
            if dones[0]:
                print(f"\nEpisode finished at step {i+1}")
                obs = env.reset()
                state=None
    
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    
    sys.stdout = originStdout

    print("The web server will shut down.")
    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Class of agent to visualize")
    parser.add_argument("--path", '-p', type=str, help='The path to the folder containing the trained agent')
    parser.add_argument("--sleep", '-s', type=float, default=0.5, help='Sleep time between steps')
    parser.add_argument("-T", type=int, default=int(1e4), help='Number of timesteps to render')
    parser.add_argument("--print-steps", action="store_true", help="Print step info to console (always saved to log)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    visualize(args.path, ALGO, args.T, args.sleep, args.print_steps)
    print('Done.')
