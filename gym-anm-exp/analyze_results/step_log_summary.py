#!/usr/bin/env python3
"""
Summarize the JSONL step log produced by rl_agents.step_logger_callback.StepLoggerCallback.

Outputs an episode-level CSV (one row per episode end per env) with totals/means for:
- reward
- objective cost components
- constraint costs
- pfe convergence rate (if present)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=str, help="Path to step_log.jsonl")
    p.add_argument("--out", type=str, default=None, help="Output CSV path (default: alongside jsonl)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if args.out is None:
        out_csv = jsonl_path.with_suffix("").with_name(jsonl_path.stem + "_episodes.csv")
    else:
        out_csv = Path(args.out)

    # episode accumulators per env_idx
    acc: Dict[int, Dict[str, Any]] = {}

    def _get_acc(env_idx: int) -> Dict[str, Any]:
        if env_idx not in acc:
            acc[env_idx] = {
                "episode_idx": 0,
                "reward_sum": 0.0,
                "steps": 0,
                "objective_costs_sum": None,
                "constraint_costs_sum": None,
                "pfe_ok_steps": 0,
                "objective_dict_sums": {},
            }
        return acc[env_idx]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as f, out_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = None

        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            env_idx = int(row.get("env_idx", 0))
            done = bool(row.get("done", False))
            reward = _as_float(row.get("reward", 0.0)) or 0.0

            a = _get_acc(env_idx)
            a["reward_sum"] += reward
            a["steps"] += 1

            obj_dict = row.get("objective", {})
            for k, v in obj_dict.items():
                val = _as_float(v)
                if val is not None:
                    a["objective_dict_sums"][k] = a["objective_dict_sums"].get(k, 0.0) + val

            # objective_costs: list of floats
            obj = row.get("objective_costs", None)
            if isinstance(obj, list):
                obj_arr = np.asarray(obj, dtype=np.float32)
                if a["objective_costs_sum"] is None:
                    a["objective_costs_sum"] = np.zeros_like(obj_arr)
                if a["objective_costs_sum"].shape == obj_arr.shape:
                    a["objective_costs_sum"] += obj_arr

            cons = row.get("constraint_costs", None)
            if isinstance(cons, list):
                cons_arr = np.asarray(cons, dtype=np.float32)
                if a["constraint_costs_sum"] is None:
                    a["constraint_costs_sum"] = np.zeros_like(cons_arr)
                if a["constraint_costs_sum"].shape == cons_arr.shape:
                    a["constraint_costs_sum"] += cons_arr

            # pfe convergence is currently logged only if env puts it in info; StepLogger stores it nowhere directly,
            # but it will appear in `objective` if it uses objective/ prefix. If it exists, use it.
            pfe = row.get("pfe_converged", None)
            if pfe is None:
                pfe = row.get("objective", {}).get("pfe_converged", None)
            if pfe is not None and bool(pfe):
                a["pfe_ok_steps"] += 1

            if done:
                ep_idx = a["episode_idx"]
                steps = int(a["steps"])

                out_row: Dict[str, Any] = {
                    "env_idx": env_idx,
                    "episode_idx": ep_idx,
                    "steps": steps,
                    "reward_sum": a["reward_sum"],
                    "reward_mean": a["reward_sum"] / max(steps, 1),
                    "pfe_ok_frac": (a["pfe_ok_steps"] / max(steps, 1)) if steps > 0 else 1.0,
                }

                if a["objective_costs_sum"] is not None:
                    for j, v in enumerate(a["objective_costs_sum"].tolist()):
                        out_row[f"objective_costs_sum_{j}"] = float(v)
                        out_row[f"objective_costs_mean_{j}"] = float(v) / max(steps, 1)

                for k, v in a["objective_dict_sums"].items():
                    k_safe = k.replace("/", "_")
                    out_row[f"{k_safe}_sum"] = float(v)
                    out_row[f"{k_safe}_mean"] = float(v) / max(steps, 1)
                if a["constraint_costs_sum"] is not None:
                    for j, v in enumerate(a["constraint_costs_sum"].tolist()):
                        out_row[f"constraint_costs_sum_{j}"] = float(v)
                        out_row[f"constraint_costs_mean_{j}"] = float(v) / max(steps, 1)

                if writer is None:
                    fieldnames = list(out_row.keys())
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(out_row)

                # reset episode accumulator
                a["episode_idx"] += 1
                a["reward_sum"] = 0.0
                a["steps"] = 0
                a["objective_costs_sum"] = None
                a["constraint_costs_sum"] = None
                a["pfe_ok_steps"] = 0
                a["objective_dict_sums"] = {}

    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

