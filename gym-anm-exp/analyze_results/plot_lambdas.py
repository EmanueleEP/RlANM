import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from . import plt_setup  # Import plotting configurations

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    parser = argparse.ArgumentParser(description="Plot Lagrangian Multipliers (lambda) over time.")
    parser.add_argument("log_dir", type=str, help="Directory containing step_log.jsonl")
    parser.add_argument("--out", type=str, default="figures/lagrangian_lambdas.pdf", help="Output plot path")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    args = parser.parse_args()

    log_path = Path(args.log_dir) / "step_log.jsonl"
    if not log_path.exists():
        print(f"Error: Could not find {log_path}")
        return

    timesteps = []
    lambda_v, lambda_p, lambda_q = [], [], []
    
    # We will sample one lambda reading per episode to avoid massive arrays
    current_ep_timesteps = []
    current_ep_lambda_v = []
    current_ep_lambda_p = []
    current_ep_lambda_q = []

    print(f"Parsing {log_path}...")
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            if "timestep" not in data or "lagrangian" not in data:
                continue
                
            lagrangian = data["lagrangian"]
            if not lagrangian:
                continue # Skip standard RL runs where lagrangian dict is empty
                
            current_ep_timesteps.append(data["timestep"])
            current_ep_lambda_v.append(lagrangian.get("lagrangian/lambda_v", 0.0))
            current_ep_lambda_p.append(lagrangian.get("lagrangian/lambda_p", 0.0))
            current_ep_lambda_q.append(lagrangian.get("lagrangian/lambda_q", 0.0))
            
            # Aggregate per episode when "done" is True
            if data.get("done", False):
                if current_ep_timesteps:
                    timesteps.append(np.mean(current_ep_timesteps))
                    lambda_v.append(np.mean(current_ep_lambda_v))
                    lambda_p.append(np.mean(current_ep_lambda_p))
                    lambda_q.append(np.mean(current_ep_lambda_q))
                
                # Reset for next episode
                current_ep_timesteps = []
                current_ep_lambda_v = []
                current_ep_lambda_p = []
                current_ep_lambda_q = []

    if not timesteps:
        print("No lagrangian data found in the logs! Make sure this was a Lagrangian run.")
        return

    # Convert to appropriate arrays and scale timesteps to Millions
    timesteps = np.array(timesteps) / 1_000_000
    lambda_v = np.array(lambda_v)
    lambda_p = np.array(lambda_p)
    lambda_q = np.array(lambda_q)

    # Smooth the data
    w = min(args.window, len(timesteps))
    # If the window consumes the whole array, reduce it so we get a line (min 2 points)
    if w == len(timesteps) and w > 1:
        w = max(1, len(timesteps) - 1)
        
    smooth_t = timesteps[w-1:] if w > 1 else timesteps
    smooth_lv = moving_average(lambda_v, w)
    smooth_lp = moving_average(lambda_p, w)
    smooth_lq = moving_average(lambda_q, w)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use markers for very short runs so single points are visible
    m = 'o' if len(smooth_t) < 5 else None
    ax.plot(smooth_t, smooth_lv, label=r'$\lambda_{voltage}$', color='#e74c3c', linewidth=2, marker=m)
    ax.plot(smooth_t, smooth_lp, label=r'$\lambda_{Active Power (P)}$', color='#3498db', linewidth=2, marker=m)
    ax.plot(smooth_t, smooth_lq, label=r'$\lambda_{Reactive Power (Q)}$', color='#2ecc71', linewidth=2, marker=m)
    ax.set_title("Evolution of Lagrangian Multipliers (Penalties)")
    ax.set_xlabel("Training Steps (Millions)")
    ax.set_ylabel(r"Penalty Multiplier Value ($\lambda$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add a note explaining lambda behavior
    plt.figtext(0.15, 0.02, 
                r"Rising $\lambda$ indicates the agent is violating constraints and the penalty is increasing." + "\n" +
                r"Flat $\lambda$ indicates the agent has learned to respect the constraint completely.",
                fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    # Adjust bottom to make room for text
    plt.subplots_adjust(bottom=0.15)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved lambda evolution plot to: {out_path}")

if __name__ == "__main__":
    main()
