# EvalCallback

import os
import warnings
import copy
from typing import Any, Dict, Optional, Union
from tqdm.auto import tqdm

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, sync_envs_normalization

from .evaluation import evaluate_policy


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar, initial=0):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self._pbar.n = initial  

    def _on_step(self):
        self._pbar.n = self.num_timesteps 
        self._pbar.update(0)
        return True

class ProgressBarManager(object):
    def __init__(self, total_timesteps, initial=0):
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.initial = initial

    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps, initial=self.initial)
        return ProgressBarCallback(self.pbar, initial=self.initial)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

"""
Evaluation callback: can be called after n. iteration to evaluate the agent and save the best model 
"""
class EvalCallback(EventCallback):

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "Pass just one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations.npz")
        self.log_path = log_path

        self.evaluations_results = []
        self.evaluation_discounted_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.evaluations_constraint_any = []
        self.evaluations_constraint_v = []
        self.evaluations_constraint_p = []
        self.evaluations_constraint_q = []
        self.evaluations_constraint_branch_overload = []
        self.evaluations_pfe_nonconverged = []
        self.evaluations_branch_overload_mva = []
        self.evaluations_branch_loading_pct = []

    def _init_callback(self) -> None:
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                f"Training and eval env are not of the same type: "
                f"{self.training_env} != {self.eval_env}"
            )
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    @staticmethod
    def _snapshot_vecnormalize(env: VecEnv) -> Optional[dict]:

        if not isinstance(env, VecNormalize):
            return None
        return {
            "obs_rms": copy.deepcopy(env.obs_rms),
            "ret_rms": copy.deepcopy(env.ret_rms),
        }

    @staticmethod
    def _restore_vecnormalize(env: VecEnv, snapshot: Optional[dict]) -> None:
        if snapshot is None or not isinstance(env, VecNormalize):
            return
        env.obs_rms = snapshot["obs_rms"]
        env.ret_rms = snapshot["ret_rms"]

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        sync_envs_normalization(self.training_env, self.eval_env)

        eval_stats_snapshot = self._snapshot_vecnormalize(self.eval_env)

        episode_rewards, episode_lengths, episode_discounted_rewards, constraint_stats = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            return_constraint_stats=True,
        )

        self._restore_vecnormalize(self.eval_env, eval_stats_snapshot)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluation_discounted_results.append(episode_discounted_rewards)
            self.evaluations_length.append(episode_lengths)
            self.evaluations_constraint_any.append(constraint_stats.get("any", []))
            self.evaluations_constraint_v.append(constraint_stats.get("v", []))
            self.evaluations_constraint_p.append(constraint_stats.get("p", []))
            self.evaluations_constraint_q.append(constraint_stats.get("q", []))
            self.evaluations_constraint_branch_overload.append(constraint_stats.get("branch_overload", []))
            self.evaluations_pfe_nonconverged.append(constraint_stats.get("pfe_nonconverged", []))
            self.evaluations_branch_overload_mva.append(constraint_stats.get("branch_overload_mva", []))
            self.evaluations_branch_loading_pct.append(constraint_stats.get("branch_loading_pct", []))
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                discounted_results=self.evaluation_discounted_results,
                ep_lengths=self.evaluations_length,
                constraint_any_steps=self.evaluations_constraint_any,
                constraint_v_steps=self.evaluations_constraint_v,
                constraint_p_steps=self.evaluations_constraint_p,
                constraint_q_steps=self.evaluations_constraint_q,
                branch_overload_steps=self.evaluations_constraint_branch_overload,
                pfe_nonconverged_steps=self.evaluations_pfe_nonconverged,
                branch_overload_mva=self.evaluations_branch_overload_mva,
                branch_loading_pct=self.evaluations_branch_loading_pct,
            )

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_discounted_reward = np.mean(episode_discounted_rewards)
        std_discounted_reward = np.std(episode_discounted_rewards)
        mean_ep_length = np.mean(episode_lengths)
        std_ep_length = np.std(episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose > 0:
            print(
                f"Evaluation: num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                f"episode_discounted_reward={mean_discounted_reward:.2f} +/- {std_discounted_reward:.2f}"
            )
            print(f"Evaluation: episode length={mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        self.logger.record("eval/mean_ep_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/mean_ep_discounted_reward", float(mean_discounted_reward))

        try:
            any_frac = np.mean(
                [a / max(1, l) for a, l in zip(constraint_stats.get("any", []), episode_lengths)]
            )
            br_frac = np.mean(
                [a / max(1, l) for a, l in zip(constraint_stats.get("branch_overload", []), episode_lengths)]
            )
            self.logger.record("eval/constraint_any_frac", float(any_frac))
            self.logger.record("eval/branch_overload_frac", float(br_frac))
            mean_br_overload_mva = np.mean(constraint_stats.get("branch_overload_mva", [0.0]))
            mean_br_loading_pct = np.mean(constraint_stats.get("branch_loading_pct", [100.0]))
            self.logger.record("eval/mean_branch_overload_mva", float(mean_br_overload_mva))
            self.logger.record("eval/mean_branch_loading_pct", float(mean_br_loading_pct))
        except Exception:
            pass

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))

                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env.save(
                        os.path.join(self.best_model_save_path, "training_vec_env")
                    )
                elif hasattr(self.eval_env, "save"):
                    self.eval_env.save(
                        os.path.join(self.best_model_save_path, "training_vec_env")
                    )

            self.best_mean_reward = mean_reward
            if self.callback is not None:
                return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        if self.callback:
            self.callback.update_locals(locals_)