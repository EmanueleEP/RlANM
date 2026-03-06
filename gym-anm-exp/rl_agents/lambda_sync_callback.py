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

    def __init__(self, vec_env, verbose=0):
        super().__init__(verbose)
        self._wrappers = []
        for e in vec_env.envs:
            w = _find_lagrangian(e)
            if w is None:
                raise ValueError("Could not find LagrangianWrapper in a sub-env.")
            self._wrappers.append(w)

    def _on_step(self) -> bool:
        if any(self.locals.get("dones", [])):
            lambdas = np.stack([w.get_lambda() for w in self._wrappers], axis=0)
            avg_lam = lambdas.mean(axis=0)
            for w in self._wrappers:
                w.set_lambda(avg_lam)
            if self.verbose >= 1:
                print(f"[LambdaSync] synced lambda = {avg_lam}")
        return True