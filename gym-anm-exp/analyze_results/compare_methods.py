#!/usr/bin/env python3
"""
Compare multiple RL runs visually to highlight the safety/performance trade-offs.
Focuses on comparing Lagrangian vs. Standard RL approaches.

Usage:
    python -m analyze_results.compare_methods <RESULT_DIR_1> <RESULT_DIR_2> ... --labels "Lagrangian" "Baseline"
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from analyze_results.plt_setup import set_rc_params

# Configure plotting aesthetics
set_rc_params()
FIGSIZE = (12, 10)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories containing step_log.jsonl files")
    parser.add_argument("--labels", nargs="+", help="Labels for the legend (matches order of dirs)")
    parser.add_argument("--out", type=str, default="figures/lagrangian_comparison.pdf", help="Output plot path")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window for episode data")
    return parser.parse_args()

def load_episodes(jsonl_path: Path):
    """Parses jsonl and aggregates data into episodes."""
    episodes = []
    current_reward = 0.0
    current_violations = 0.0
    current_steps = 0
    
    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found.")
        return None

    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("type") == "meta":
                    continue
                
                # Use environment reward (pre-normalization if possible)
                reward = data.get("reward_env")
                if reward is None:
                    reward = data.get("reward", 0.0)
                
                current_reward += float(reward)
                
                # Check for any physical constraint violation
                costs = data.get("constraint_costs", [0, 0, 0])
                # We count the magnitude of violations
                current_violations += sum(max(0, c) for c in costs)
                
                current_steps += 1
                
                if data.get("done", False):
                    episodes.append({
                        "reward": current_reward,
                        "violations": current_violations,
                        "steps": current_steps,
                        "timestep": data.get("timestep", 0)
                    })
                    current_reward = 0.0
                    current_violations = 0.0
                    current_steps = 0
            except Exception:
                continue
                
    return episodes

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    args = parse_args()
    labels = args.labels if args.labels else [Path(d).name for d in args.dirs]
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIGSIZE, sharex=False)
    plt.subplots_adjust(hspace=0.4)

    safety_stats = []

    for i, (run_dir, label) in enumerate(zip(args.dirs, labels)):
        jsonl_path = Path(run_dir) / "step_log.jsonl"
        data = load_episodes(jsonl_path)
        
        if not data:
            continue
            
        timesteps = np.array([e["timestep"] for e in data]) / 1e6
        rewards = np.array([e["reward"] for e in data])
        violations = np.array([e["violations"] for e in data])
        
        # 1. Convergence plot (smoothed)
        smooth_rewards = moving_average(rewards, args.window)
        smooth_x = timesteps[args.window-1:]
        ax1.plot(smooth_x, smooth_rewards, label=label, linewidth=2)
        
        # 2. Safety Magnitude plot (smoothed)
        smooth_violations = moving_average(violations, args.window)
        ax2.plot(smooth_x, smooth_violations, label=label, linewidth=2)
        
        # 3. Frequency calculation for the bar chart
        safe_episodes = np.sum(violations < 1e-4)
        safe_frac = (safe_episodes / len(violations)) * 100
        safety_stats.append((label, safe_frac))

    # Formatting axes
    ax1.set_title("Economic Performance (Episode Reward Sum)")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Safety Violations Magnitude (Lower is better)")
    ax2.set_ylabel("Sum of Violations / Episode")
    ax2.set_xlabel("Timesteps (Millions)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Set log scale if violations vary wildly
    if any(s[1] > 0 for s in safety_stats):
         ax2.set_yscale('symlog', linthresh=0.1)

    # 4. Violation Frequency Bar Chart
    if safety_stats:
        bar_labels, bar_values = zip(*safety_stats)
        bars = ax3.bar(bar_labels, bar_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'][:len(bar_labels)])
        ax3.set_title("Reliability (% of Perfectly Safe Episodes)")
        ax3.set_ylabel("% Safe Steps")
        ax3.set_ylim(0, 110)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Comparison plot saved to: {args.out}")

if __name__ == "__main__":
    main()
