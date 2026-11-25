"""
Evaluate an MPC policy on external (real) profiles with the same episodes and
metrics as analyze_results.evaluate, so that MPC and RL results are comparable.

`perfect` reads the future loads and generation from the profile bank installed
by rl_agents.profiles, or uses the upstream MPCAgentPerfect on ANM6Easy's
built-in daily profiles when no profile file is given; `constant` repeats the
current values.

Example:
    python -m mpc_policies.evaluate_mpc perfect \
        --profile-file '../../Datasets/processed/profiles_{split}.npz' \
        --split test -N 32 --safety-margin 0.96 --out results/mpc/perfect_N32
"""

import argparse
import os

import numpy as np
from gym_anm import MPCAgentConstant, MPCAgentPerfect
from gym_anm.agents.mpc import MPCAgent
from stable_baselines3.common.vec_env import DummyVecEnv

from analyze_results.evaluate import make_eval_env, run_episodes, print_summary, save_results
from rl_agents.hyperparameters import GAMMA, MAX_EVAL_EP_LENGTH
from rl_agents.utils import get_training_config


class MPCAgentProfilePerfect(MPCAgent):
    """MPC with perfect forecasts taken from an external profile bank."""

    def forecast(self, env):
        bank = env.external_profile_bank
        scenario, timestep = env._external_profile_position
        length = bank.load_p_mw.shape[1]
        steps = (timestep + 1 + np.arange(self.planning_steps)) % length
        P_load_forecast = bank.load_p_mw[scenario, steps].T / self.baseMVA
        P_gen_forecast = bank.gen_p_max_mw[scenario, steps].T / self.baseMVA
        return P_load_forecast, P_gen_forecast


POLICIES = {"constant": MPCAgentConstant, "perfect": MPCAgentProfilePerfect}


def evaluate_mpc(policy, planning_steps, safety_margin, n_episodes, seed,
                 profile_file, split, eval_profile_seed=456,
                 env_id="Custom_ANM6Easy-v0", max_ep_length=MAX_EVAL_EP_LENGTH):
    env_config = get_training_config()
    env = DummyVecEnv([make_eval_env(
        env_id, max_ep_length, env_config, eval_profile_seed,
        profile_file=profile_file, profile_split=split,
        battery_clip_penalty=0.0,
    )])
    base = env.envs[0].unwrapped
    agent_class = POLICIES[policy]
    if policy == "perfect" and profile_file is None:
        agent_class = MPCAgentPerfect
    agent = agent_class(base.simulator, base.action_space, GAMMA,
                             safety_margin=safety_margin, planning_steps=planning_steps)
    low, high = base.action_space.low, base.action_space.high

    def act(obs, state):
        action = agent.act(base)
        # The wrapper stack expects actions normalized to [-1, 1].
        return (2.0 * (action - low) / (high - low) - 1.0)[None, :], state

    episodes = run_episodes(env, act, n_episodes, GAMMA, seed)
    env.close()
    return episodes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", choices=tuple(POLICIES))
    parser.add_argument("--profile-file", default=None,
                        help="NPZ/CSV profiles, {split} is replaced by --split "
                             "(default: ANM6Easy built-in profiles).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--planning-steps", "-N", type=int, default=32)
    parser.add_argument("--safety-margin", type=float, default=0.96)
    parser.add_argument("--episodes", "-n", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_EVAL_EP_LENGTH,
                        help="Episode length (same default as RL evaluation).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Same meaning and default as analyze_results.evaluate.")
    parser.add_argument("--eval-profile-seed", type=int, default=456)
    parser.add_argument("--out", required=True, help="Directory for evaluation/eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    label = f"MPC-{args.policy} N={args.planning_steps} beta={args.safety_margin}"
    episodes = evaluate_mpc(
        args.policy, args.planning_steps, args.safety_margin, args.episodes, args.seed,
        os.path.abspath(os.path.expanduser(args.profile_file)) if args.profile_file else None,
        args.split,
        eval_profile_seed=args.eval_profile_seed, max_ep_length=args.max_steps,
    )
    print_summary(episodes, label)
    os.makedirs(args.out, exist_ok=True)
    save_results(episodes, args.out, label)


if __name__ == "__main__":
    main()
