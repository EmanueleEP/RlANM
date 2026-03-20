import json
import os
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _to_jsonable(x: Any, *, max_list_elems: int = 200) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, (list, tuple)):
        if len(x) > max_list_elems:
            return {"type": "list", "len": len(x), "head": [_to_jsonable(v) for v in x[:10]]}
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if x.size <= max_list_elems:
            return x.tolist()
        return {
            "type": "ndarray",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        }
    return str(x)


class StepLoggerCallback(BaseCallback):
    """
    Writes a JSONL step log with key training signals:
    - timestep, env index, action, reward, done
    - objective reward components (from env info)
    - constraint_costs + lagrangian penalty/lambda if enabled
    - observation (either full list for small arrays or summary stats)
    """

    def __init__(
        self,
        log_path: str,
        log_observations: bool = True,
        log_actions: bool = True,
        log_every: int = 1,
        flush_every: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_path = log_path
        self.log_observations = bool(log_observations)
        self.log_actions = bool(log_actions)
        self.log_every = int(log_every)
        if self.log_every <= 0:
            raise ValueError("log_every must be >= 1")
        self.flush_every = int(flush_every)
        self._f = None
        self._n_written = 0
        # Use a delta-based trigger (not modulo), because `num_timesteps` may
        # increase by `n_envs` each rollout step and may never hit exact multiples
        # of `log_every` (e.g. n_envs=6, log_every=1000).
        self._last_logged_timestep = -1

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # Line-buffered so you can `tail -f` the file and see updates quickly,
        # even when `log_every` is large and logs are infrequent.
        self._f = open(self.log_path, "a", encoding="utf-8", buffering=1)
        # Write a small marker so the file is created and visibly non-empty even
        # when `log_every` is large and episodes are long.
        self._f.write(json.dumps({"type": "meta", "event": "training_start", "timestep": int(self.num_timesteps)}) + "\n")
        self._f.flush()
        self._last_logged_timestep = int(self.num_timesteps)

    def _on_training_end(self) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def _on_step(self) -> bool:
        if self._f is None:
            return True

        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        actions = self.locals.get("actions", None)
        obs = self.locals.get("new_obs", self.locals.get("obs", None))

        # Log periodically, but also log on episode boundaries so you always get some data
        # even when `log_every` is large.
        done_any = False
        if isinstance(dones, (list, tuple, np.ndarray)):
            done_any = bool(np.any(dones))
        elif dones is not None:
            done_any = bool(dones)

        # Log when enough SB3 timesteps have elapsed since the last log, but also
        # always log on episode boundaries.
        if (not done_any) and (int(self.num_timesteps) - int(self._last_logged_timestep) < self.log_every):
            return True
        self._last_logged_timestep = int(self.num_timesteps)

        # VecEnv case: arrays/lists per env
        n_envs = 1
        if isinstance(infos, (list, tuple)):
            n_envs = len(infos)

        for i in range(n_envs):
            info_i: Dict[str, Any] = infos[i] if isinstance(infos, (list, tuple)) else (infos or {})
            reward_i = rewards[i] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
            done_i = dones[i] if isinstance(dones, (list, tuple, np.ndarray)) else dones
            action_i = actions[i] if isinstance(actions, (list, tuple, np.ndarray)) else actions
            obs_i = obs[i] if isinstance(obs, (list, tuple, np.ndarray)) and n_envs > 1 else obs

            row: Dict[str, Any] = {
                "timestep": int(self.num_timesteps),
                "env_idx": int(i),
                # NOTE: `reward` here is whatever SB3 sees (may be normalized/clipped by VecNormalize).
                "reward": _to_jsonable(reward_i),
                # Best-effort raw env rewards (before VecNormalize). When LagrangianWrapper is enabled,
                # it provides `original_reward` and `lagrangian_penalty`.
                "reward_env": _to_jsonable(info_i.get("original_reward", None)),
                "reward_lagrangian": _to_jsonable(
                    (float(info_i["original_reward"]) - float(info_i["lagrangian_penalty"]))
                    if ("original_reward" in info_i and "lagrangian_penalty" in info_i)
                    else None
                ),
                "done": _to_jsonable(done_i),
                "pfe_converged": _to_jsonable(info_i.get("pfe_converged", None)),
                "constraint_costs": _to_jsonable(info_i.get("constraint_costs", None)),
                "constraint": _to_jsonable({k: v for k, v in info_i.items() if str(k).startswith("constraint/")}),
                "objective_costs": _to_jsonable(info_i.get("objective_costs", None)),
                "objective": _to_jsonable({k: v for k, v in info_i.items() if str(k).startswith("objective/")}),
                "lagrangian": _to_jsonable({k: v for k, v in info_i.items() if str(k).startswith("lagrangian/")}),
            }
            if self.log_actions:
                row["action"] = _to_jsonable(action_i)
            if self.log_observations:
                row["obs"] = _to_jsonable(obs_i)

            self._f.write(json.dumps(row) + "\n")
            self._n_written += 1

        # Ensure progress is visible on disk for infrequent logging (e.g. every 1000 steps).
        if self.log_every >= 100:
            self._f.flush()
        elif self.flush_every > 0 and (self._n_written % self.flush_every) == 0:
            self._f.flush()

        return True
