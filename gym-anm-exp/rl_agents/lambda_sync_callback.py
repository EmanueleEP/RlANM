import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .wrappers import LagrangianWrapper


def _find_lagrangian(env):
    # return the LagrangianWrapper or None
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
    - Each environment trains with lambda_lr (no individual updates)
    - This callback averages lambda values across all environments at episode boundaries
    - The averaged lambda is then pushed back to all environments
    
    This ensures stable, synchronized dual variable updates without race conditions.
    """

    def __init__(self, vec_env, dual_lr: float, verbose=0):
        super().__init__(verbose)
        self.dual_lr = float(dual_lr)
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
            if any(dones):
                # Collect effective costs for envs that finished this step.
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
                        # Peak-aware blend (same formula as LagrangianWrapper._update_lambda)
                        effective = (1.0 - self.peak_weight) * avg_arr + self.peak_weight * peak_arr
                    else:
                        # Fallback: avg only (old behaviour, e.g. if info key is missing)
                        effective = avg_arr
    
                    episode_effective.append(effective)
    
                if episode_effective:
                    avg_effective = np.stack(episode_effective, axis=0).mean(axis=0)
                    cost_limit = np.asarray(self._wrappers[0].cost_limit, dtype=np.float32)
                    gradient = avg_effective - cost_limit
    
                    lambdas = np.stack([w.get_lambda() for w in self._wrappers], axis=0)
                    avg_lam = lambdas.mean(axis=0)
                    new_lam = np.clip(avg_lam + self.dual_lr * gradient, 0.0, self._wrappers[0].lambda_max)
    
                    for w in self._wrappers:
                        w.set_lambda(new_lam)
    
                    if self.verbose >= 1:
                        print(
                            f"[LambdaSync] effective={avg_effective} "
                            f"limit={cost_limit} grad={gradient} lambda={new_lam}"
                        )
            return True