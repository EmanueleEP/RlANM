"""
Aggregate test-set evaluations of several methods into one table.

Each group is LABEL=GLOB, where GLOB matches run directories that contain
evaluation/eval_results.json (written by analyze_results.evaluate or
mpc_policies.evaluate_mpc). For every run the metrics are averaged over its
episodes; the table reports mean +/- std across the runs of a group (seeds).

Example:
    python -m analyze_results.compare_test \
        --group "MPC perfect=results/mpc/perfect_N32" \
        --group "PPO=results/experiment_01/original/seed_*" \
        --group "PPO Lagrangian=results/experiment_01/lagrangian/seed_*" \
        --group "PPO Lagrangian PID=results/experiment_01/lagrangian_pid/seed_*" \
        --out results/experiment_01/test_comparison
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

METRICS = [
    ("Reward gym-anm", "reward", ".1f"),
    ("Disc. reward gym-anm", "discounted_reward", ".1f"),
    ("Steps with violations (%)", "any_violation_frac", ".1%"),
    ("Voltage violations (%)", "v_violation_frac", ".1%"),
    ("Power/branch violations (%)", "p_violation_frac", ".1%"),
    ("Branch overload (%)", "branch_overload_frac", ".1%"),
    ("Voltage viol. max (pu)", "v_violation_mag_max", ".4f"),
    ("Power/branch viol. max (pu)", "p_violation_mag_max", ".4f"),
    ("PFE non-convergence", "pfe_nonconverged", ".1f"),
    ("Curtailment (MWh)", "curtailment_total_mwh", ".1f"),
    ("Grid import (MWh)", "slack_import_total_mwh", ".1f"),
    ("Grid export (MWh)", "slack_export_total_mwh", ".1f"),
    ("Battery SoC mean (MWh)", "battery_soc_mean", ".1f"),
]


def load_group(pattern):
    runs = []
    for run in sorted(glob.glob(pattern)):
        path = os.path.join(run, "evaluation", "eval_results.json")
        if not os.path.isfile(path):
            print(f"WARNING: no evaluation in {run}")
            continue
        with open(path) as file:
            episodes = json.load(file)["episodes"]
        runs.append({key: float(np.mean([e[key] for e in episodes])) for _, key, _ in METRICS})
    if not runs:
        raise FileNotFoundError(f"No evaluated runs match {pattern!r}")
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB")
    parser.add_argument("--out", required=True, help="Output prefix for .md and .csv")
    args = parser.parse_args()

    groups = []
    for spec in args.group:
        label, separator, pattern = spec.partition("=")
        if not separator:
            parser.error(f"--group must be LABEL=GLOB, got {spec!r}")
        groups.append((label, load_group(pattern)))

    header = "| Metric | " + " | ".join(f"{label} (n={len(runs)})" for label, runs in groups) + " |"
    lines = [header, "|" + "---|" * (len(groups) + 1)]
    rows = []
    for name, key, fmt in METRICS:
        cells = []
        for label, runs in groups:
            values = np.array([run[key] for run in runs])
            mean, std = values.mean(), values.std()
            cells.append(f"{mean:{fmt}} ± {std:{fmt}}")
            rows.append({"metric": key, "method": label, "mean": mean, "std": std, "n_runs": len(runs)})
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".md", "w") as file:
        file.write(table + "\n")
    with open(args.out + ".csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "method", "mean", "std", "n_runs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
