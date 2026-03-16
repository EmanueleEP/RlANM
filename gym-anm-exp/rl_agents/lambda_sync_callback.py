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
            # Collect avg costs for envs that ended this step.
            episode_avgs = []
            for i, done in enumerate(dones):
                if not done or i >= len(infos):
                    continue
                avg = infos[i].get("lagrangian/episode_cost_avg", None)
                if avg is None:
                    continue
                episode_avgs.append(np.asarray(avg, dtype=np.float32))

            # Sync/update lambda only when we have the signals.
            if episode_avgs:
                avg_cost = np.stack(episode_avgs, axis=0).mean(axis=0)
                cost_limit = np.asarray(self._wrappers[0].cost_limit, dtype=np.float32)
                gradient = avg_cost - cost_limit

                lambdas = np.stack([w.get_lambda() for w in self._wrappers], axis=0)
                avg_lam = lambdas.mean(axis=0)
                new_lam = avg_lam + self.dual_lr * gradient

                for w in self._wrappers:
                    w.set_lambda(new_lam)

                if self.verbose >= 1:
                    print(f"[LambdaSync] avg_cost={avg_cost} limit={cost_limit} grad={gradient} lambda={new_lam}")
        return True
