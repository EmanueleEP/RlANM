"""
Callback for monitoring battery exploration during training.
"""
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BatteryExplorationCallback(BaseCallback):
    """
    Callback that monitors battery usage during training and provides logging.

    Tracks both the raw policy action (what the agent *intended*) and the
    physical outcome after BatteryClipWrapper (what the simulator *executed*).
    Comparing the two gives a direct measure of how often the battery is being
    clipped by SoC or power limits.

    Usage:
        - Logs battery intent vs. outcome every 1000 timesteps when verbose=1
        - Helps verify that the agent is learning to use the battery effectively
        - Provides insights into charging/discharging patterns and clip frequency
    """

    def __init__(self, verbose: int = 0, battery_reward_scale: float = 0.1):
        """
        Args:
            verbose: Verbosity level (0 = silent, 1 = print stats every 1000 steps)
            battery_reward_scale: Unused; kept for API compatibility.
        """
        super().__init__(verbose)
        self.battery_reward_scale = battery_reward_scale

        _maxlen = 1000  # only last 1000 entries are needed for logging
        self.raw_action_history: deque = deque(maxlen=_maxlen)
        self.intended_mw_history: deque = deque(maxlen=_maxlen)
        self.battery_usage_history: deque = deque(maxlen=_maxlen)
        self.soc_history: deque = deque(maxlen=_maxlen)

        # Cached values (populated on first step)
        self._p_min_mw: float = -50.0
        self._p_max_mw: float = 50.0
        self._battery_device_id: int = 3   # fallback for BCP
        self._battery_action_idx: int = 2  # fallback for BCP
        self._limits_cached: bool = False

    def _get_base_env(self):
        """Unwrap VecEnv and Gymnasium wrappers to reach the base ANM env."""
        vec = self.training_env
        while hasattr(vec, "venv"):
            vec = vec.venv
        env = vec.envs[0] if hasattr(vec, "envs") and vec.envs else vec
        while hasattr(env, "env"):
            env = env.env
        return env

    def _cache_limits(self, base_env) -> None:
        """Read battery device id, action index, and P limits from the simulator once."""
        if self._limits_cached:
            return
        try:
            from gym_anm.simulator.components import Generator, StorageUnit
            sim = base_env.simulator
            if sim.baseMVA <= 0:
                return

            # Find StorageUnit device id dynamically
            battery_dev_id = None
            for dev_id, dev in sim.devices.items():
                if isinstance(dev, StorageUnit):
                    battery_dev_id = dev_id
                    break
            if battery_dev_id is None:
                return

            self._battery_device_id = battery_dev_id
            self._p_min_mw = float(sim.devices[battery_dev_id].p_min) * float(sim.baseMVA)
            self._p_max_mw = float(sim.devices[battery_dev_id].p_max) * float(sim.baseMVA)

            # Compute action index: layout is [P_gen...][Q_gen...][P_des...][Q_des...]
            n_gen = sum(
                1 for dev in sim.devices.values()
                if isinstance(dev, Generator) and not dev.is_slack
            )
            self._battery_action_idx = 2 * n_gen

            self._limits_cached = True
        except Exception:
            pass  # keep fallback values, try again next step

    def _rescale_to_mw(self, normalized_action: float) -> float:
        """
        Convert a normalized action in [-1, 1] to physical MW using the same
        formula as NormalizeActionWrapper.rescale_action():
            MW = p_min + 0.5 * (a + 1) * (p_max - p_min)
        """
        return self._p_min_mw + 0.5 * (normalized_action + 1.0) * (self._p_max_mw - self._p_min_mw)

    def _on_step(self) -> bool:
        """Called at each training step to monitor battery usage."""

        base_env = self._get_base_env()

        # Cache battery limits on first call
        if hasattr(base_env, "simulator"):
            self._cache_limits(base_env)

        # Record raw policy intent and rescale to MW
        raw_battery_action = None
        actions = self.locals.get("actions", None)
        if actions is not None and np.asarray(actions).shape[-1] > self._battery_action_idx:
            raw_battery_action = float(np.asarray(actions)[0][self._battery_action_idx])
            intended_mw = self._rescale_to_mw(raw_battery_action)
            self.raw_action_history.append(raw_battery_action)
            self.intended_mw_history.append(intended_mw)

        # Record physical outcome from simulator
        if hasattr(base_env, "simulator"):
            full_state = base_env.simulator.state
            dev_p   = full_state.get("dev_p",  {}).get("MW",  {})
            des_soc = full_state.get("des_soc", {}).get("MWh", {})

            battery_p   = float(dev_p.get(self._battery_device_id,   0.0))
            battery_soc = float(des_soc.get(self._battery_device_id, 0.0))

            self.battery_usage_history.append(battery_p)
            self.soc_history.append(battery_soc)

        # Periodic statistics
        if self.verbose > 0 and self.num_timesteps % 1000 == 0:

            if self.battery_usage_history:
                avg_physical = np.mean(np.abs(list(self.battery_usage_history)))
                avg_soc      = np.mean(list(self.soc_history)) if self.soc_history else 0.0
                print(
                    f"[BatteryExploration] t={self.num_timesteps} | "
                    f"Avg |P| (physical): {avg_physical:.3f} MW | "
                    f"Avg SoC: {avg_soc:.3f} MWh"
                )

            if self.intended_mw_history and self.battery_usage_history:
                avg_intended = np.mean(np.abs(list(self.intended_mw_history)))
                avg_physical = np.mean(np.abs(list(self.battery_usage_history)))

                if avg_intended > 1e-3:
                    clip_ratio = 1.0 - avg_physical / avg_intended
                else:
                    clip_ratio = 0.0  # agent isn't asking for much — nothing to clip

                print(
                    f"[BatteryExploration] t={self.num_timesteps} | "
                    f"Avg |intended| (MW): {avg_intended:.3f} | "
                    f"Avg |physical| (MW): {avg_physical:.3f} | "
                    f"Clip ratio: {clip_ratio:.1%}"
                )

        return True