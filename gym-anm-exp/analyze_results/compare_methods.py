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
    """Load exact episode summaries produced by StepLoggerCallback."""
    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found. Skipping this directory.")
        return None

    episodes = []
    legacy_rows = 0
    malformed_rows = 0

    with open(jsonl_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                malformed_rows += 1
                continue
            if not isinstance(data, dict):
                malformed_rows += 1
                continue

            row_type = data.get("type")
            if row_type == "episode_summary":
                try:
                    reward = float(data["reward"])
                    steps = int(data["steps"])
                    violations = data.get("violations")
                    violations = np.nan if violations is None else float(violations)
                    timestep = int(data.get("timestep", 0))
                    if not np.isfinite(reward) or steps <= 0:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    print(f"Warning: invalid episode summary at {jsonl_path}:{line_number}; skipped.")
                    continue
                episodes.append({
                    "reward": reward,
                    "violations": violations,
                    "steps": steps,
                    "timestep": timestep,
                })
            elif row_type not in ("meta", "step_sample"):
                legacy_rows += 1

    if malformed_rows:
        print(f"Warning: {malformed_rows} malformed rows in {jsonl_path} were skipped.")
    if not episodes and legacy_rows:
        print(
            f"Warning: {jsonl_path} is a legacy sampled log and has no exact episode summaries. "
            "It cannot be used for episode totals; generate new data with the updated logger."
        )
    return episodes

def moving_average(x, w):
    if w <= 0:
        raise ValueError("Smoothing window must be >= 1")
    effective_window = min(len(x), w)
    if effective_window == 0:
        return np.asarray(x), effective_window
    values = np.convolve(x, np.ones(effective_window), 'valid') / effective_window
    return values, effective_window

def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.dirs):
        raise ValueError("--labels must contain exactly one label for each directory")
    labels = args.labels if args.labels else [Path(d).name for d in args.dirs]
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIGSIZE, sharex=False)

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
        smooth_rewards, effective_window = moving_average(rewards, args.window)
        smooth_x = timesteps[effective_window - 1:]
        ax1.plot(smooth_x, smooth_rewards, label=label, linewidth=2)
        
        # 2. Safety Magnitude plot (smoothed)
        valid = np.isfinite(violations)
        if np.any(valid):
            valid_timesteps = timesteps[valid]
            smooth_violations, violation_window = moving_average(violations[valid], args.window)
            ax2.plot(valid_timesteps[violation_window - 1:], smooth_violations,
                     label=label, linewidth=2)

            # 3. Frequency calculation for episodes with complete constraint data
            safe_episodes = np.sum(violations[valid] < 1e-4)
            safe_frac = (safe_episodes / np.sum(valid)) * 100
            safety_stats.append((label, safe_frac))

    # Formatting axes
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Training Timesteps (millions)")
    ax1.set_ylabel("Cumulative Reward")
    if ax1.lines:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Constraint Violations per Episode")
    ax2.set_xlabel("Training Timesteps (millions)")
    ax2.set_ylabel("Total Violation Magnitude")
    if ax2.lines:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Use symlog if violation range spans more than 2 orders of magnitude
    if safety_stats and any(s[1] < 100 for s in safety_stats):
        ax2.set_yscale('symlog', linthresh=0.1)

    # 4. Violation Frequency Bar Chart
    if safety_stats:
        bar_labels, bar_values = zip(*safety_stats)
        bars = ax3.bar(bar_labels, bar_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'][:len(bar_labels)])
        ax3.set_title("Safety Rate")
        ax3.set_ylabel("Safe Episodes (%)")
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
