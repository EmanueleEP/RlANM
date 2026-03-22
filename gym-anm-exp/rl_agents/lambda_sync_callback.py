import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .wrappers import LagrangianWrapper


def _find_lagrangian(env):
    """Return the LagrangianWrapper in the wrapper stack, or None."""
    inner = env
    while inner is not None:
        if isinstance(inner, LagrangianWrapper):
            return inner
        inner = getattr(inner, "env", None)
    return None


class LambdaSyncCallback(BaseCallback):
    """
    Synchronizes Lagrange multipliers (lambda) across multiple parallel environments.

    When using multiple environments with the Lagrangian method:
    - Each environment trains with lambda_lr=0 (no individual updates)
    - This callback averages lambda values across all environments at episode boundaries
    - The averaged lambda is then pushed back to all environments

    This ensures stable, synchronized dual variable updates without race conditions.
    """

    def __init__(self, vec_env, dual_lr: float, peak_weight: float = 0.02, verbose: int = 0):
        super().__init__(verbose)
        self.dual_lr = float(dual_lr)
        self.peak_weight = float(peak_weight)
        self._wrappers = []

        inner = vec_env
        while hasattr(inner, "venv"):
            inner = inner.venv

        for e in inner.envs:
            w = _find_lagrangian(e)
            if w is None:
                raise ValueError("Could not find LagrangianWrapper in a sub-env.")
            self._wrappers.append(w)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if not any(dones):
            return True

        episode_effective = []
        for i, done in enumerate(dones):
            if not done or i >= len(infos):
                continue

            avg  = infos[i].get("lagrangian/episode_cost_avg", None)
            peak = infos[i].get("lagrangian/episode_cost_max", None)

            if avg is None:
                continue

            avg_arr = np.asarray(avg, dtype=np.float32)

            if peak is not None:
                peak_arr = np.asarray(peak, dtype=np.float32)
                effective = (
                    (1.0 - self.peak_weight) * avg_arr
                    + self.peak_weight * peak_arr
                )
            else:
                effective = avg_arr

            episode_effective.append(effective)

        if episode_effective:
            avg_eff = np.stack(episode_effective, axis=0).mean(axis=0)
            cost_limit = np.asarray(self._wrappers[0].cost_limit, dtype=np.float32)
            gradient = avg_eff - cost_limit

            avg_lam = np.stack(
                [w.get_lambda() for w in self._wrappers], axis=0
            ).mean(axis=0)

            new_lam = np.clip(
                avg_lam + self.dual_lr * gradient,
                0.0,
                self._wrappers[0].lambda_max,
            )

            for w in self._wrappers:
                w.set_lambda(new_lam)

            if self.verbose >= 1:
                print(
                    f"[LambdaSync] eff={avg_eff} "
                    f"limit={cost_limit} "
                    f"grad={gradient} "
                    f"lam={new_lam}"
                )

        return True