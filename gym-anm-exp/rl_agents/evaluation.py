"""
This file contains a custom `evaluate_policy()` function, which is a modified
version of the Stable Baselines3 original function
(https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html#stable_baselines3.common.evaluation.evaluate_policy )
which also computes discounted returns.
"""
from typing import Callable, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv

from .hyperparameters import GAMMA

def _get_gamma(env: Union[gym.Env, VecEnv]) -> float:
    """
    Extract gamma (discount factor) from environment.
    Falls back to GAMMA from hyperparameters if not found.
    """
    # Prova a ottenere gamma direttamente
    if hasattr(env, 'gamma'):
        return env.gamma
    # Se VecEnv, esamina il primo env interno
    if isinstance(env, VecEnv) and hasattr(env, 'envs') and len(env.envs) > 0:
        inner_env = env.envs[0]
        while hasattr(inner_env, 'env'):
            if hasattr(inner_env, 'gamma'):
                return inner_env.gamma
            inner_env = inner_env.env
        if hasattr(inner_env, 'gamma'):
            return inner_env.gamma
    # Default importato da hyperparameters
    return GAMMA

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    return_constraint_stats: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean..
    :param return_constraint_stats: If True, also returns per-episode constraint/violation counters
        derived from the env `info` dict (works with or without LagrangianWrapper).
    :return: Mean reward per episode, std of reward per episode, mean discounted reward per episode, std of discounted
        reward per episode. Returns ([float], [int], [float]) when ``return_episode_rewards`` is True.
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    # Get gamma once before the loop
    gamma = _get_gamma(env)

    episode_rewards, episode_lengths, episode_discounted_rewards = [], [], []
    constraint_any_steps, constraint_v_steps, constraint_p_steps, constraint_q_steps = [], [], [], []
    branch_overload_steps, pfe_nonconverged_steps = [], []
    branch_overload_mva_episodes, branch_loading_pct_episodes = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        any_steps = v_steps = p_steps = q_steps = 0
        br_steps = 0
        pfe_steps = 0
        branch_overload_mag_sum = 0.0
        branch_loading_pct_sum = 0.0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)

            # Extract the data for the single environment (index 0)
            reward = rewards[0]
            done = dones[0] 
            # The 'done' flag from VecEnv is True if terminated OR truncated,

            episode_reward += reward
            episode_discounted_reward += reward * (gamma ** episode_length)  # <-- USE EXTRACTED GAMMA

            # --- Constraint/violation accounting (optional) ---
            if return_constraint_stats:
                info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
                try:
                    if isinstance(info0, dict):
                        c = info0.get("constraint_costs", None)
                        if c is not None:
                            c = np.asarray(c, dtype=float).flatten()
                            if c.size >= 3:
                                if float(c[0]) > 0.0:
                                    v_steps += 1
                                if float(c[1]) > 0.0:
                                    p_steps += 1
                                if float(c[2]) > 0.0:
                                    q_steps += 1
                                if (float(c[0]) > 0.0) or (float(c[1]) > 0.0) or (float(c[2]) > 0.0):
                                    any_steps += 1

                        br = info0.get("constraint/branch_overload_pu_max", None)
                        if br is not None:
                            br_val = float(br)
                            branch_overload_mag_sum += br_val
                            if br_val > 0.0:
                                br_steps += 1

                        bl = info0.get("branch_loading_pu", None)
                        if bl is not None and isinstance(bl, dict) and bl:
                            branch_loading_pct_sum += max(bl.values()) * 100.0

                        pfe = info0.get("pfe_converged", None)
                        if pfe is False:
                            pfe_steps += 1
                except Exception:
                    pass

            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_discounted_rewards.append(episode_discounted_reward)
        if return_constraint_stats:
            constraint_any_steps.append(int(any_steps))
            constraint_v_steps.append(int(v_steps))
            constraint_p_steps.append(int(p_steps))
            constraint_q_steps.append(int(q_steps))
            branch_overload_steps.append(int(br_steps))
            pfe_nonconverged_steps.append(int(pfe_steps))
            branch_overload_mva_episodes.append(branch_overload_mag_sum / max(episode_length, 1) * 100.0)
            branch_loading_pct_episodes.append(branch_loading_pct_sum / max(episode_length, 1))
    mean_reward = np.mean(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_reward = np.std(episode_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        if return_constraint_stats:
            stats = {
                "any": constraint_any_steps,
                "v": constraint_v_steps,
                "p": constraint_p_steps,
                "q": constraint_q_steps,
                "branch_overload": branch_overload_steps,
                "pfe_nonconverged": pfe_nonconverged_steps,
                "branch_overload_mva": branch_overload_mva_episodes,
                "branch_loading_pct": branch_loading_pct_episodes,
            }
            return episode_rewards, episode_lengths, episode_discounted_rewards, stats
        return episode_rewards, episode_lengths, episode_discounted_rewards
    return mean_reward, std_reward, mean_discounted_reward, std_discounted_reward
