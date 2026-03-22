"""
Callback for monitoring battery exploration during training.
"""
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

    # Index of the battery P action in the action vector (device 3 = battery).
    BATTERY_ACTION_IDX = 2

    def __init__(self, verbose: int = 0, battery_reward_scale: float = 0.1):
        """
        Args:
            verbose: Verbosity level (0 = silent, 1 = print stats every 1000 steps)
            battery_reward_scale: Unused; kept for API compatibility.
        """
        super().__init__(verbose)
        self.battery_reward_scale = battery_reward_scale

        self.raw_action_history: list = []    # policy output [-1, 1] for battery P
        self.intended_mw_history: list = []   # raw action rescaled to MW (pre-clip intent)
        self.battery_usage_history: list = [] # physical P injection (MW) after clipping
        self.soc_history: list = []           # battery SoC (MWh)

        # Battery P limits in MW — read from the simulator on first step and
        # cached so we don't re-traverse the wrapper stack every step.
        self._p_min_mw: float = -50.0
        self._p_max_mw: float = 50.0
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
        """Read battery P limits from the simulator once and cache them."""
        if self._limits_cached:
            return
        try:
            sim = base_env.simulator
            dev = sim.devices.get(3, None)
            if dev is not None and sim.baseMVA > 0:
                self._p_min_mw = float(dev.p_min) * float(sim.baseMVA)
                self._p_max_mw = float(dev.p_max) * float(sim.baseMVA)
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

        # 1. Cache battery limits on first call
        if hasattr(base_env, "simulator"):
            self._cache_limits(base_env)

        # 2. Record raw policy intent and rescale to MW
        raw_battery_action = None
        actions = self.locals.get("actions", None)
        if actions is not None and np.asarray(actions).shape[-1] > self.BATTERY_ACTION_IDX:
            raw_battery_action = float(np.asarray(actions)[0][self.BATTERY_ACTION_IDX])
            intended_mw = self._rescale_to_mw(raw_battery_action)
            self.raw_action_history.append(raw_battery_action)
            self.intended_mw_history.append(intended_mw)

        # 3. Record physical outcome from simulator
        if hasattr(base_env, "simulator"):
            full_state = base_env.simulator.state
            dev_p   = full_state.get("dev_p",  {}).get("MW",  {})
            des_soc = full_state.get("des_soc", {}).get("MWh", {})

            battery_p   = float(dev_p.get(3,   0.0))
            battery_soc = float(des_soc.get(3, 0.0))

            self.battery_usage_history.append(battery_p)
            self.soc_history.append(battery_soc)

        # 4. Periodic statistics
        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            window = 100

            if self.battery_usage_history:
                avg_physical = np.mean(np.abs(self.battery_usage_history[-window:]))
                avg_soc      = np.mean(self.soc_history[-window:]) if self.soc_history else 0.0
                print(
                    f"[BatteryExploration] t={self.num_timesteps} | "
                    f"Avg |P| (physical): {avg_physical:.3f} MW | "
                    f"Avg SoC: {avg_soc:.3f} MWh"
                )

            if self.intended_mw_history and self.battery_usage_history:
                avg_intended = np.mean(np.abs(self.intended_mw_history[-window:]))
                avg_physical = np.mean(np.abs(self.battery_usage_history[-window:]))

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