"""
This script can be used to visualize a trained-agent interacting
with an environment, assuming the environment supports rendering.

Usage
-----
python -m analyze_results.visualize <ALGO> -p <PATH> -T <TIMESTEPS>
"""
import os
import time
import argparse
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from rl_agents.hyperparameters import ENV_ID
from rl_agents.utils import load_visualization_env


def visualize(path, algo, T, sleep_time):
    """
    Visualize a trained agent.
    
    gym-anm environments render in a web browser. The environment will start
    a local web server and you should open the provided URL in your browser.
    """
    # Load agent and environment
    model = algo.load(os.path.join(path, 'best_model'))
    env = load_visualization_env(ENV_ID, os.path.join(path, 'training_vec_env'), 1)

    print("IMPORTANT: gym-anm environments render in a WEB BROWSER")
    print("'Running on http://127.0.0.1:8050/'")
    print("\nStarting visualization in 3 seconds")
    time.sleep(3)
    
    # Reset and take first step to initialize rendering
    obs = env.reset()
    
    # Call render to start the web server
    print("\nInitializing web server...")
    base_env = env.envs[0]
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # First render call starts the server
    base_env.render()
    
    print("If you see a URL above, open it in your Windows browser!")
    print("\nStarting agent interaction...")
    
    # Run the agent
    done, state = False, None
    
    try:
        for i in range(T):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
            # Render after each step
            base_env.render()
            
            print(f"Step {i+1}/{T}, Reward: {rewards[0]:.3f}", end='\r')
            
            time.sleep(sleep_time)
            
            # Check if episode is done
            if dones[0]:
                print(f"\nEpisode finished at step {i+1}")
                obs = env.reset()
    
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    
    print("\n\nClosing environment...")
    print("The web server will shut down.")
    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Class of agent to visualize")
    parser.add_argument("--path", '-p', type=str, help='The path to the folder containing the trained agent')
    parser.add_argument("--sleep", '-s', type=float, default=0.5, help='Sleep time between steps')
    parser.add_argument("-T", type=int, default=int(1e4), help='Number of timesteps to render')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.agent == 'PPO':
        ALGO = PPO
    elif args.agent == 'SAC':
        ALGO = SAC
    else:
        raise ValueError('Unimplemented agent ' + args.agent)

    visualize(args.path, ALGO, args.T, args.sleep)
    print('Done.')