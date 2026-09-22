"""
Evaluate a trained RL agent on the BCP environment.

Usage:
    python -m analyze_results.evaluate PPO --path results/BCP-v0/.../run_X --episodes 20
    python -m analyze_results.evaluate PPO --path run1 run2 --labels "Lagrangian" "Standard" --episodes 20
"""

import argparse
import os
import json
import warnings
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_agents.hyperparameters import (
    GAMMA,
    BATTERY_CLIP_PENALTY,
    ENV_ID as DEFAULT_ENV_ID,
    MAX_EVAL_EP_LENGTH as DEFAULT_MAX_EVAL_EP_LENGTH,
    V_TOLERANCE,
    PQ_LIMIT,
    Q_LIMIT,
    PEAK_WEIGHT,
    LAMBDA_MAX,
    PFE_TERMINATE_PENALTY,
    PFE_LAMBDA_STEPS,
    FORECAST_HORIZON_STEPS,
)

from rl_agents.utils import (
    make_env_with_profile_seed,
    configure_env,
    get_training_config,
    load_run_config,
)

def read_run_hyperparameters(run_path):
    """Read ENV_ID and MAX_EVAL_EP_LENGTH from a run's hyperparameters.txt."""
    hp_path = os.path.join(run_path, "hyperparameters.txt")
    params = {}
    if not os.path.exists(hp_path):
        return params
    with open(hp_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            key = parts[0].strip().strip("'\"")
            val = parts[1].strip().strip("'\"")
            params[key] = val
    return params
from rl_agents.wrappers import (
    TimeLimitWrapper, NormalizeActionWrapper, BatteryClipWrapper,
    PFETerminateWrapper, LagrangianWrapper, ForecastObsWrapper,
)


def make_eval_env(env_id, max_ep_length, env_config,
                  eval_profile_seed: int = 456,
                  solar_scale: float = 1.0, load_scale: float = 1.0,
                  profile_file=None, profile_provider=None,
                  profile_split="validation",
                  lambda_values=None, v_tolerance: float = V_TOLERANCE,
                  pq_limit: float = PQ_LIMIT, q_limit: float = Q_LIMIT,
                  peak_weight: float = PEAK_WEIGHT,
                  lambda_max: float = LAMBDA_MAX,
                  pfe_penalty: float = PFE_TERMINATE_PENALTY,
                  pfe_lambda_steps: float = PFE_LAMBDA_STEPS,
                  forecast_horizon: int = FORECAST_HORIZON_STEPS,
                  battery_clip_penalty: float = BATTERY_CLIP_PENALTY):
    """Build an eval env, optionally reproducing the frozen training reward."""
    def _init():
        env = make_env_with_profile_seed(
            env_id,
            profile_seed=eval_profile_seed,
            solar_scale=solar_scale,
            load_scale=load_scale,
            profile_file=profile_file,
            profile_provider=profile_provider,
            profile_split=profile_split,
        )
        reproduce_lagrangian = lambda_values is not None
        configure_env(env, env_config, use_lagrangian_reward=reproduce_lagrangian)
        env = TimeLimitWrapper(env, max_episode_steps=max_ep_length)
        env = NormalizeActionWrapper(env)
        env = BatteryClipWrapper(env, clip_penalty_weight=battery_clip_penalty if reproduce_lagrangian else 0.0)
        if forecast_horizon > 0:
            env = ForecastObsWrapper(env, horizon_steps=forecast_horizon)
        if reproduce_lagrangian:
            env = LagrangianWrapper(
                env,
                v_tolerance=v_tolerance,
                pq_limit=pq_limit,
                q_limit=q_limit,
                lambda_lr=0.0,
                lambda_init=lambda_values,
                lambda_max=lambda_max,
                peak_weight=peak_weight,
            )
        env = PFETerminateWrapper(
            env,
            penalty=pfe_penalty,
            lambda_steps=pfe_lambda_steps if reproduce_lagrangian else 0.0,
        )
        return env
    return _init


def load_lambda_state(run_path):
    """Load the Lagrangian multipliers paired with the best model."""
    lam_path = os.path.join(run_path, "best_lambda_state.npy")
    if os.path.exists(lam_path):
        lam = np.asarray(np.load(lam_path, allow_pickle=False), dtype=np.float32)
        if lam.shape != (3,) or not np.all(np.isfinite(lam)) or np.any(lam < 0):
            raise ValueError(f"Invalid lambda state in {lam_path}: expected 3 finite, non-negative values")
        return lam
    return None


def evaluate_agent(run_path, algo_class, n_episodes, gamma, seed=42,
                   eval_profile_seed=456, solar_scale=1.0, load_scale=1.0,
                   env_id_override=None, profile_file_override=None,
                   profile_provider_override=None, profile_split_override=None):
    """Run n_episodes and collect detailed metrics."""

    vec_env_path = os.path.join(run_path, "training_vec_env")
    zip_path = os.path.join(run_path, "best_model.zip")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model not found: {zip_path}. Training may not have completed.")

    # Read ENV_ID and MAX_EVAL_EP_LENGTH from run's hyperparameters
    hp = read_run_hyperparameters(run_path)
    env_id = env_id_override if env_id_override else hp.get("ENV_ID", DEFAULT_ENV_ID)
    max_ep_length = int(hp.get("MAX_EVAL_EP_LENGTH", DEFAULT_MAX_EVAL_EP_LENGTH))
    use_lagrangian = hp.get("USE_LAGRANGIAN", "False").lower() == "true"
    # Saved run values (environment reward/constraint parameters included),
    # falling back to the current hyperparameters.py for older runs.
    env_config = load_run_config(run_path, get_training_config())

    # Load lambda for Lagrangian reward computation
    lam = load_lambda_state(run_path) if use_lagrangian else None
    if lam is not None:
        print(f"  Lambda loaded: {lam}")
    elif use_lagrangian:
        print(f"  WARNING: USE_LAGRANGIAN=True but no best_lambda_state.npy found")

    print(f"  ENV_ID: {env_id}, MAX_EVAL_EP_LENGTH: {max_ep_length}")

    # Build env
    def saved_float(key, default):
        value = hp.get(key)
        return default if value in (None, "None") else float(value)

    def saved_optional(key):
        value = hp.get(key)
        return None if value in (None, "", "None") else value

    if profile_file_override is not None or profile_provider_override is not None:
        profile_file = profile_file_override
        profile_provider = profile_provider_override
    else:
        profile_file = saved_optional("EVAL_PROFILE_FILE")
        profile_provider = saved_optional("EVAL_PROFILE_PROVIDER")
    profile_split = profile_split_override or saved_optional("EVAL_PROFILE_SPLIT") or "validation"

    env = DummyVecEnv([make_eval_env(
        env_id,
        max_ep_length,
        env_config,
        eval_profile_seed,
        solar_scale,
        load_scale,
        profile_file=profile_file,
        profile_provider=profile_provider,
        profile_split=profile_split,
        lambda_values=lam,
        v_tolerance=saved_float("V_TOLERANCE", V_TOLERANCE),
        pq_limit=saved_float("PQ_LIMIT", PQ_LIMIT),
        q_limit=saved_float("Q_LIMIT", Q_LIMIT),
        peak_weight=saved_float("PEAK_WEIGHT", PEAK_WEIGHT),
        lambda_max=saved_float("LAMBDA_MAX", LAMBDA_MAX),
        pfe_penalty=saved_float("PFE_TERMINATE_PENALTY", PFE_TERMINATE_PENALTY),
        pfe_lambda_steps=saved_float("PFE_LAMBDA_STEPS", PFE_LAMBDA_STEPS),
        forecast_horizon=int(saved_float("FORECAST_HORIZON_STEPS", FORECAST_HORIZON_STEPS)),
        battery_clip_penalty=float(env_config["BATTERY_CLIP_PENALTY"]),
    )])

    if os.path.exists(vec_env_path):
        env = VecNormalize.load(vec_env_path, env)
        env.training = False
        env.norm_reward = False
    else:
        warnings.warn(f"VecNormalize not found at {vec_env_path}")

    # Load model
    model = algo_class.load(zip_path, env=env, device=device,
                            custom_objects={"learning_rate": 3e-4, "clip_range": 0.2})

    episodes = []

    for ep in range(n_episodes):
        # VecEnv.seed() schedules the seed for the next reset. Offset it per
        # episode to obtain a reproducible sequence of distinct trajectories.
        env.seed(seed + ep)
        obs = env.reset()
        done = False
        state = None

        ep_reward = 0.0
        ep_discounted_reward = 0.0
        ep_lagrangian_reward = 0.0
        ep_lagrangian_discounted_reward = 0.0
        ep_lagrangian_penalty_total = 0.0
        ep_steps = 0
        discount = 1.0

        # Per-step accumulators
        v_violation_steps = 0
        p_violation_steps = 0
        q_violation_steps = 0
        any_violation_steps = 0
        branch_overload_steps = 0
        pfe_nonconverged = 0

        # Continuous metrics
        battery_soc_list = []
        battery_power_list = []
        curtailment_power_list = []
        curtailment_energy_list = []
        slack_import_list = []
        slack_export_list = []
        soc_boundary_list = []
        branch_soft_list = []
        q_magnitude_list = []
        anm_penalty_list = []   # ANM6Easy penalty component (lamb*violations)

        # Per-step violation magnitudes (only when violation occurs, to assess severity)
        v_magnitude_list = []       # raw pu voltage violation when > 0
        p_magnitude_list = []       # raw pu P/branch violation when > 0
        q_violation_magnitude_list = []  # raw pu reactive-power violation when > 0
        branch_overload_mag_list = []  # raw pu branch overload when > 0
        v_constraint_cost_list = []  # normalized costs consumed by the wrappers
        p_constraint_cost_list = []
        q_constraint_cost_list = []

        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            training_r = float(reward[0])
            raw_r = float(info.get("reward_base_anm", training_r))

            ep_reward += raw_r
            ep_discounted_reward += discount * raw_r
            ep_steps += 1

            # With lambda available, the environment stack already reproduces
            # the training reward. Keep the ANM reward above as a separate metric.
            if lam is not None:
                lag_penalty = float(info.get("lagrangian_penalty", 0.0))
                lag_r = training_r
            else:
                lag_penalty = 0.0
                lag_r = raw_r
            ep_lagrangian_reward += lag_r
            ep_lagrangian_discounted_reward += discount * lag_r
            ep_lagrangian_penalty_total += lag_penalty

            discount *= gamma

            # Constraint costs
            costs = np.asarray(info.get("constraint_costs", [0, 0, 0]), dtype=np.float32)
            v_constraint_cost_list.append(float(costs[0]))
            p_constraint_cost_list.append(float(costs[1]))
            q_constraint_cost_list.append(float(costs[2]))

            # Physical violation magnitudes in pu. These are distinct from the
            # normalized constraint_costs used by the learning wrappers.
            v_magnitude = float(info.get("constraint/v_violation_pu_max", 0.0))
            p_magnitude = float(info.get("constraint/combined_p_violation_pu_max", 0.0))
            q_violation_magnitude = float(info.get("constraint/q_violation_pu_max", 0.0))
            if v_magnitude > 1e-6:
                v_violation_steps += 1
                v_magnitude_list.append(v_magnitude)
            if p_magnitude > 1e-6:
                p_violation_steps += 1
                p_magnitude_list.append(p_magnitude)
            if q_violation_magnitude > 1e-6:
                q_violation_steps += 1
                q_violation_magnitude_list.append(q_violation_magnitude)
            if max(v_magnitude, p_magnitude, q_violation_magnitude) > 1e-6:
                any_violation_steps += 1

            # Branch overload
            br_over = float(info.get("constraint/branch_overload_pu_max", 0.0))
            if br_over > 1e-6:
                branch_overload_steps += 1
                branch_overload_mag_list.append(br_over)

            # PFE
            if info.get("pfe_terminated", False):
                pfe_nonconverged += 1

            # Objective components
            battery_soc_list.append(info.get("objective/battery_soc_mwh", 0.0))
            curtailment_power_list.append(info.get("objective/curtailment_mw", 0.0))
            curtailment_energy_list.append(info.get("objective/cost_curtailment_mwh", 0.0))
            slack_import_list.append(info.get("objective/cost_slack_import_mwh", 0.0))
            slack_export_list.append(info.get("objective/cost_slack_export_mwh", 0.0))
            soc_boundary_list.append(info.get("objective/cost_soc_boundary", 0.0))
            branch_soft_list.append(info.get("objective/cost_branch_soft_mvah", 0.0))
            q_magnitude_list.append(info.get("objective/cost_q_magnitude_mvarh", 0.0))

            # ANM6Easy reward decomposition: reward_base_anm = -(e_loss + penalty)
            # penalty = lamb * violations (lamb=100 in BCP). High penalty variance = instability.
            anm_r = info.get("reward_base_anm", None)
            if anm_r is not None:
                # e_loss ≥ 0, penalty ≥ 0  →  reward_base_anm = -(e_loss + penalty)
                # We proxy penalty as: reward_base_anm - (-e_loss_proxy)
                # Direct: just track the raw reward per step (before VecNormalize)
                anm_penalty_list.append(float(anm_r))

            # Battery power (from dev_p of storage device)
            slack_p = info.get("objective/slack_p_mw", 0.0)
            battery_power_list.append(slack_p)  # will replace below if available

            done = dones[0]

        ep_data = {
            "episode": ep,
            "reward": ep_reward,
            "discounted_reward": ep_discounted_reward,
            "has_lambda": lam is not None,
            "lagrangian_reward": ep_lagrangian_reward,
            "lagrangian_discounted_reward": ep_lagrangian_discounted_reward,
            "lagrangian_penalty_total": ep_lagrangian_penalty_total,
            "steps": ep_steps,
            # Violations (step counts)
            "v_violation_steps": v_violation_steps,
            "p_violation_steps": p_violation_steps,
            "q_violation_steps": q_violation_steps,
            "any_violation_steps": any_violation_steps,
            "branch_overload_steps": branch_overload_steps,
            "pfe_nonconverged": pfe_nonconverged,
            # Violation fractions
            "v_violation_frac": v_violation_steps / ep_steps,
            "p_violation_frac": p_violation_steps / ep_steps,
            "q_violation_frac": q_violation_steps / ep_steps,
            "any_violation_frac": any_violation_steps / ep_steps,
            "branch_overload_frac": branch_overload_steps / ep_steps,
            # Battery
            "battery_soc_mean": float(np.mean(battery_soc_list)),
            "battery_soc_min": float(np.min(battery_soc_list)),
            "battery_soc_max": float(np.max(battery_soc_list)),
            "battery_soc_std": float(np.std(battery_soc_list)),
            # Costs
            "curtailment_mean_mw": float(np.mean(curtailment_power_list)),
            "curtailment_total_mwh": float(np.sum(curtailment_energy_list)),
            "slack_import_total_mwh": float(np.sum(slack_import_list)),
            "slack_export_total_mwh": float(np.sum(slack_export_list)),
            "soc_boundary_total": float(np.sum(soc_boundary_list)),
            "branch_soft_total_mvah": float(np.sum(branch_soft_list)),
            "q_magnitude_total_mvarh": float(np.sum(q_magnitude_list)),
            # Per-step reward stats: std reveals how much the penalty spikes
            "anm_reward_per_step_mean": float(np.mean(anm_penalty_list)) if anm_penalty_list else 0.0,
            "anm_reward_per_step_std":  float(np.std(anm_penalty_list))  if anm_penalty_list else 0.0,
            "anm_reward_per_step_min":  float(np.min(anm_penalty_list))  if anm_penalty_list else 0.0,
            # Violation magnitude (when violation occurs): assess severity, not just frequency
            "v_violation_mag_mean":  float(np.mean(v_magnitude_list))  if v_magnitude_list  else 0.0,
            "v_violation_mag_max":   float(np.max(v_magnitude_list))   if v_magnitude_list  else 0.0,
            "p_violation_mag_mean":  float(np.mean(p_magnitude_list))  if p_magnitude_list  else 0.0,
            "p_violation_mag_max":   float(np.max(p_magnitude_list))   if p_magnitude_list  else 0.0,
            "q_violation_mag_mean":  float(np.mean(q_violation_magnitude_list)) if q_violation_magnitude_list else 0.0,
            "q_violation_mag_max":   float(np.max(q_violation_magnitude_list))  if q_violation_magnitude_list else 0.0,
            "branch_overload_mag_mean": float(np.mean(branch_overload_mag_list)) if branch_overload_mag_list else 0.0,
            "branch_overload_mag_max":  float(np.max(branch_overload_mag_list))  if branch_overload_mag_list else 0.0,
            # Dimensionless normalized costs used by the learning wrappers.
            "v_constraint_cost_mean": float(np.mean(v_constraint_cost_list)),
            "v_constraint_cost_max": float(np.max(v_constraint_cost_list)),
            "p_constraint_cost_mean": float(np.mean(p_constraint_cost_list)),
            "p_constraint_cost_max": float(np.max(p_constraint_cost_list)),
            "q_constraint_cost_mean": float(np.mean(q_constraint_cost_list)),
            "q_constraint_cost_max": float(np.max(q_constraint_cost_list)),
        }
        episodes.append(ep_data)

        safe = "SAFE" if any_violation_steps == 0 else f"{any_violation_steps} violations"
        print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f}, "
              f"disc={ep_discounted_reward:.1f}, steps={ep_steps}, {safe}")

    env.close()
    return episodes


def print_summary(episodes, label=""):
    """Print aggregate statistics across episodes."""
    n = len(episodes)
    header = f"  EVALUATION SUMMARY — {label}" if label else "  EVALUATION SUMMARY"
    print(header)

    def stat(key):
        vals = [e[key] for e in episodes]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    # Reward
    mu, sd, mn, mx = stat("reward")
    print(f"\n  Reward (raw):             {mu:.1f} +/- {sd:.1f}  (min={mn:.1f}, max={mx:.1f})")
    mu, sd, mn, mx = stat("discounted_reward")
    print(f"  Reward (raw discounted):  {mu:.1f} +/- {sd:.1f}  (min={mn:.1f}, max={mx:.1f})")

    # Lagrangian reward (only if non-zero)
    lag_rewards = [e["lagrangian_reward"] for e in episodes]
    if any(abs(r - e["reward"]) > 1e-6 for r, e in zip(lag_rewards, episodes)):
        mu, sd, mn, mx = stat("lagrangian_reward")
        print(f"Reward (lagrangian): {mu:.1f} +/- {sd:.1f}  (min={mn:.1f}, max={mx:.1f})")
        mu, sd, mn, mx = stat("lagrangian_discounted_reward")
        print(f"Reward (lagr. disc.): {mu:.1f} +/- {sd:.1f}  (min={mn:.1f}, max={mx:.1f})")
        mu, sd, _, _ = stat("lagrangian_penalty_total")
        print(f"Lagrangian penalty total: {mu:.1f} +/- {sd:.1f}")

    # Episode length
    mu, sd, _, _ = stat("steps")
    print(f"Episode length: {mu:.0f} +/- {sd:.0f}")

    # Violations — frequency
    print(f"\n  Constraint Violations:")
    print(f"    {'Type':<20s}  {'Steps/ep':>10}  {'% steps':>10}  {'Var %':>10}  {'Mag mean':>10}  {'Mag max':>10}")
    print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for key, name, frac_key, mag_mean_key, mag_max_key in [
        ("any_violation_steps",   "Any",          "any_violation_frac",    None,                   None),
        ("v_violation_steps",     "Voltage",      "v_violation_frac",      "v_violation_mag_mean",  "v_violation_mag_max"),
        ("p_violation_steps",     "Power/Branch", "p_violation_frac",      "p_violation_mag_mean",  "p_violation_mag_max"),
        ("q_violation_steps",     "Reactive",     "q_violation_frac",      "q_violation_mag_mean",  "q_violation_mag_max"),
        ("branch_overload_steps", "Branch overload", "branch_overload_frac", "branch_overload_mag_mean", "branch_overload_mag_max"),
    ]:
        mu_steps, sd_steps, _, _ = stat(key)
        mu_frac, sd_frac, _, _   = stat(frac_key)
        mag_mean_str = f"{np.mean([e[mag_mean_key] for e in episodes]):.4f}" if mag_mean_key else "    —"
        mag_max_str  = f"{np.max([e[mag_max_key]  for e in episodes]):.4f}" if mag_max_key  else "    —"
        print(f"    {name:<20s}  {mu_steps:>7.1f}     {mu_frac*100:>7.1f}%   ±{sd_frac*100:>5.1f}%   {mag_mean_str:>10}  {mag_max_str:>10}")

    # Safety rate
    safe_episodes = sum(1 for e in episodes if e["any_violation_steps"] == 0)
    print(f"\n  Safe episodes: {safe_episodes}/{n} ({safe_episodes/n*100:.1f}%)")

    # PFE
    pfe_total = sum(e["pfe_nonconverged"] for e in episodes)
    print(f"PFE non-convergence: {pfe_total} total across {n} episodes")

    # Battery
    print(f"\nBattery:")
    mu, sd, _, _ = stat("battery_soc_mean")
    print(f"    Mean SoC: {mu:.1f} +/- {sd:.1f} MWh")
    mu_min, _, _, _ = stat("battery_soc_min")
    mu_max, _, _, _ = stat("battery_soc_max")
    print(f"SoC range: [{mu_min:.1f}, {mu_max:.1f}] MWh (avg min/max)")

    # Cost breakdown
    print(f"\n  Cost Breakdown (mean per episode):")
    for key, name, unit in [
        ("curtailment_total_mwh", "Curtailment", "MWh"),
        ("slack_import_total_mwh", "Grid import", "MWh"),
        ("slack_export_total_mwh", "Grid export", "MWh"),
        ("soc_boundary_total", "SoC boundary penalty", ""),
        ("branch_soft_total_mvah", "Branch soft penalty", "MVAh"),
        ("q_magnitude_total_mvarh", "Q magnitude", "MVArh"),
    ]:
        mu, sd, _, _ = stat(key)
        print(f"    {name:<25s}: {mu:8.2f} +/- {sd:6.2f} {unit}")

    # Reward-per-step variance: evidence of penalty spikes
    if any(e.get("anm_reward_per_step_std", 0.0) > 0 for e in episodes):
        mu_m, _, _, _ = stat("anm_reward_per_step_mean")
        mu_s, _, _, _ = stat("anm_reward_per_step_std")
        mu_n, _, _, _ = stat("anm_reward_per_step_min")
        print(f"Mean per step: {mu_m:8.3f}")
        print(f"Std  per step: {mu_s:8.3f}")
        print(f"Min  per step: {mu_n:8.3f}")


def save_results(episodes, run_path, label=""):
    """Save detailed results to JSON."""
    out_dir = os.path.join(run_path, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_results.json")

    output = {
        "run_path": run_path,
        "label": label,
        "n_episodes": len(episodes),
        "episodes": episodes,
        "summary": {
            "reward_mean": float(np.mean([e["reward"] for e in episodes])),
            "reward_std": float(np.std([e["reward"] for e in episodes])),
            "discounted_reward_mean": float(np.mean([e["discounted_reward"] for e in episodes])),
            "lagrangian_reward_mean": float(np.mean([e["lagrangian_reward"] for e in episodes])),
            "lagrangian_reward_std": float(np.std([e["lagrangian_reward"] for e in episodes])),
            "lagrangian_penalty_mean": float(np.mean([e["lagrangian_penalty_total"] for e in episodes])),
            "any_violation_frac_mean": float(np.mean([e["any_violation_frac"] for e in episodes])),
            "branch_overload_frac_mean": float(np.mean([e["branch_overload_frac"] for e in episodes])),
            "safe_episodes": sum(1 for e in episodes if e["any_violation_steps"] == 0),
            "battery_soc_mean": float(np.mean([e["battery_soc_mean"] for e in episodes])),
        },
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_file}")
    return out_file


def build_comparison_table(all_results):
    """Build comparison table as a list of lines (for both printing and saving)."""
    lines = []
    lines.append("  COMPARISON TABLE")

    header = f"  {'Metric':<30s}"
    for label, _ in all_results:
        header += f"  {label:>18s}"
    lines.append(header)

    sep = f"  {'-'*30}"
    for _ in all_results:
        sep += f"  {'-'*18}"
    lines.append(sep)

    metrics = [
        ("Reward (raw)", "reward", ".1f"),
        ("Disc. Reward (raw)", "discounted_reward", ".1f"),
        ("Violations (%)", "any_violation_frac", ".1%"),
        ("Branch overload (%)", "branch_overload_frac", ".1%"),
        ("Branch overload mag (pu)", "branch_overload_mag_mean", ".4f"),
        ("Voltage viol. mag (pu)", "v_violation_mag_mean", ".4f"),
        ("Power/branch viol. mag (pu)", "p_violation_mag_mean", ".4f"),
        ("Reactive viol. mag (pu)", "q_violation_mag_mean", ".4f"),
        ("Battery SoC (MWh)", "battery_soc_mean", ".1f"),
        ("Curtailment (MWh)", "curtailment_total_mwh", ".1f"),
        ("Grid import (MWh)", "slack_import_total_mwh", ".1f"),
        ("Q magnitude (MVArh)", "q_magnitude_total_mvarh", ".1f"),
    ]

    lagrangian_metrics = [
        ("Reward (lagrangian)", "lagrangian_reward", ".1f"),
        ("Lagr. penalty total", "lagrangian_penalty_total", ".1f"),
    ]

    for name, key, fmt in metrics:
        row = f"  {name:<30s}"
        for _, eps in all_results:
            mu = np.mean([e[key] for e in eps])
            sd = np.std([e[key] for e in eps])
            val_str = f"{mu:{fmt}} +/- {sd:{fmt}}"
            row += f"  {val_str:>18s}"
        lines.append(row)

    any_has_lambda = any(
        eps[0].get("has_lambda", False) for _, eps in all_results
    )
    if any_has_lambda:
        for name, key, fmt in lagrangian_metrics:
            row = f"  {name:<30s}"
            for _, eps in all_results:
                if eps[0].get("has_lambda", False):
                    mu = np.mean([e[key] for e in eps])
                    sd = np.std([e[key] for e in eps])
                    val_str = f"{mu:{fmt}} +/- {sd:{fmt}}"
                    row += f"  {val_str:>18s}"
                else:
                    row += f"  {'N/A':>18s}"
            lines.append(row)

    row = f"  {'Safe episodes':<30s}"
    for _, eps in all_results:
        safe = sum(1 for e in eps if e["any_violation_steps"] == 0)
        row += f"  {f'{safe}/{len(eps)}':>18s}"
    lines.append(row)

    return lines


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument("agent", type=str, choices=["PPO", "SAC"], help="Algorithm")
    parser.add_argument("--path", "-p", nargs="+", required=True, help="Run directory path(s)")
    parser.add_argument("--labels", nargs="+", help="Labels for each run (for comparison)")
    parser.add_argument("--episodes", "-n", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-profile-seed", type=int, default=456,
                        help="Seed for load/generation profiles. Default 456 (different from training 123).")
    parser.add_argument("--solar-scale", type=float, default=1.0,
                        help="Scale factor for solar generation profile. "
                             "1.0=normal, 1.3=high solar (summer), 0.5=low solar (winter/cloudy).")
    parser.add_argument("--load-scale", type=float, default=1.0,
                        help="Scale factor for load profile. "
                             "1.0=normal, 1.4=high load (winter), 0.8=low load (summer).")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Override ENV_ID from hyperparameters.txt. Useful to evaluate "
                             "ANM6Easy-trained agents on Custom_ANM6Easy-v0 for detailed metrics.")
    profile_source = parser.add_mutually_exclusive_group()
    profile_source.add_argument(
        "--profile-file", default=None,
        help="Override the saved evaluation profile with an NPZ or CSV file.",
    )
    profile_source.add_argument(
        "--profile-provider", default=None,
        help="Override the saved evaluation profile with package.module:callable.",
    )
    parser.add_argument(
        "--profile-split", default=None,
        help="Override the saved evaluation split.",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    algo_class = PPO if args.agent == "PPO" else SAC
    labels = args.labels if args.labels else [os.path.basename(p) for p in args.path]

    all_results = []

    for run_path, label in zip(args.path, labels):
        print(f"\nEvaluating: {label}")
        print(f"  Path: {run_path}")
        print(f"  Episodes: {args.episodes}")
        print()

        episodes = evaluate_agent(
            run_path, algo_class, args.episodes, GAMMA,
            seed=args.seed, eval_profile_seed=args.eval_profile_seed,
            solar_scale=args.solar_scale, load_scale=args.load_scale,
            env_id_override=args.env_id,
            profile_file_override=args.profile_file,
            profile_provider_override=args.profile_provider,
            profile_split_override=args.profile_split,
        )
        print_summary(episodes, label)

        if not args.no_save:
            save_results(episodes, run_path, label)

        all_results.append((label, episodes))

    # Comparison table if multiple runs
    if len(all_results) > 1:
        table_lines = build_comparison_table(all_results)
        table_text = "\n".join(table_lines)
        print(table_text)

        if not args.no_save:
            # Save to common parent directory
            common_dir = os.path.commonpath(args.path)
            if os.path.isfile(common_dir):
                common_dir = os.path.dirname(common_dir)
            table_path = os.path.join(common_dir, "comparison.txt")
            with open(table_path, "w") as f:
                f.write(table_text + "\n")
            print(f"Comparison table saved to: {table_path}")


if __name__ == "__main__":
    main()
