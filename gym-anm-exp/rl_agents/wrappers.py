import warnings
import gymnasium as gym
import numpy as np



class ForecastObsWrapper(gym.ObservationWrapper):
    """Appends a short lookahead window of exogenous variables to the observation."""

    def __init__(self, env: gym.Env, horizon_steps: int = 4):
        super().__init__(env)
        self.horizon_steps = int(horizon_steps)
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be >= 1")

        base = env.unwrapped
        if not hasattr(base, "P_loads") or not hasattr(base, "P_maxs"):
            raise ValueError("Underlying env must expose P_loads and P_maxs for ForecastObsWrapper.")

        self._p_loads = np.asarray(base.P_loads, dtype=np.float32)
        self._p_maxs = np.asarray(base.P_maxs, dtype=np.float32)
        if self._p_loads.ndim != 2 or self._p_maxs.ndim != 2:
            raise ValueError("P_loads and P_maxs must be 2D arrays.")

        self._steps_per_day = int(self._p_loads.shape[1])
        if self._p_maxs.shape[1] != self._steps_per_day:
            raise ValueError("P_loads and P_maxs must have the same number of timesteps per day.")

        added = self.horizon_steps * (self._p_loads.shape[0] + self._p_maxs.shape[0])
        old_space = self.observation_space
        if not isinstance(old_space, gym.spaces.Box) or len(old_space.shape) != 1:
            raise ValueError("ForecastObsWrapper expects a 1D Box observation space.")

        forecast_low = -1e6 * np.ones(added, dtype=np.float32)
        forecast_high = 1e6 * np.ones(added, dtype=np.float32)
        low = np.concatenate([np.asarray(old_space.low, dtype=np.float32), forecast_low])
        high = np.concatenate([np.asarray(old_space.high, dtype=np.float32), forecast_high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        base = self.env.unwrapped
        time_idx = int(getattr(base, "state", np.array([0]))[-1])
        feats = []
        for h in range(1, self.horizon_steps + 1):
            idx = (time_idx + h) % self._steps_per_day
            feats.append(self._p_loads[:, idx])
            feats.append(self._p_maxs[:, idx])
        forecast = np.concatenate(feats, axis=0).astype(np.float32)
        return np.concatenate([np.asarray(observation, dtype=np.float32), forecast], axis=0)



class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), \
            "This wrapper only works with continuous action space (spaces.Box)"
        self.low, self.high = action_space.low, action_space.high
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )
        super().__init__(env)

    def rescale_action(self, scaled_action):
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, seed=None, options=None):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class PFETerminateWrapper(gym.Wrapper):
    """Terminates episode on PFE divergence and applies a penalty.

    When a LagrangianWrapper is present in the inner wrapper stack, the penalty
    scales dynamically with lambda so that crashing PFE is never cheaper than
    surviving the episode:

        penalty = base_penalty + sum(lambda * cost_limit) * lambda_steps

    With lambda_steps ~ max_episode_length / 2, this approximates the expected
    remaining Lagrangian cost of the episode, making early termination
    unprofitable at any lambda value.

    When no LagrangianWrapper is found, falls back to the fixed base_penalty.
    """

    def __init__(self, env: gym.Env, penalty: float = 100.0, lambda_steps: float = 0.0):
        super().__init__(env)
        self.penalty = float(penalty)
        self.lambda_steps = float(lambda_steps)
        self._lagrangian: "LagrangianWrapper | None" = None
        self._lagrangian_searched = False

    def _find_lagrangian(self):
        if self._lagrangian_searched:
            return self._lagrangian
        env = self.env
        while env is not None:
            if isinstance(env, LagrangianWrapper):
                self._lagrangian = env
                break
            env = getattr(env, "env", None)
        self._lagrangian_searched = True
        return self._lagrangian

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("pfe_converged", True) is False:
            total_penalty = self.penalty
            if self.lambda_steps > 0:
                lag = self._find_lagrangian()
                if lag is not None:
                    total_penalty += float(np.dot(lag.lam, lag.cost_limit)) * self.lambda_steps
            reward = float(reward) - total_penalty
            terminated = True
            info["pfe_terminated"] = True
            info["pfe_penalty"] = total_penalty
        return obs, reward, terminated, truncated, info


class LagrangianWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        v_tolerance: float = 1.0,
        pq_limit: float = 1.0,
        q_limit: float = None,
        lambda_init=1.0,
        lambda_lr: float = 1e-2,
        normalize_by_episode_length: bool = True,
        lambda_max: float = 10.0,
        n_envs: int = 1,
        peak_weight: float = 0.2,
    ):
        super().__init__(env)
        base_env = env.unwrapped
        self.n_buses = base_env.simulator.N_bus
        self.v_tol = v_tolerance
        self.lambda_lr = lambda_lr
        self.normalize = normalize_by_episode_length
        self.lambda_max = lambda_max
        self.n_constraints = 3
        q_lim = q_limit if q_limit is not None else pq_limit
        self.cost_limit = np.array([v_tolerance, pq_limit, q_lim], dtype=np.float32)
        self.peak_weight = float(np.clip(peak_weight, 0.0, 1.0))

        # coherence check between lambda_lr and n_envs
        if n_envs > 1 and lambda_lr > 0.0:
            raise ValueError(
                f"n_envs={n_envs} > 1, use lambda_lr=0.0 and "
                "LambdaSyncCallback for updates."
            )

        lambda_init_arr = np.atleast_1d(np.array(lambda_init, dtype=np.float32))
        if lambda_init_arr.shape == (self.n_constraints,):
            self.lam = lambda_init_arr.copy()
        else:
            self.lam = np.full(self.n_constraints, float(lambda_init_arr[0]), dtype=np.float32)

        self.episode_cost = np.zeros(self.n_constraints, dtype=np.float32)
        self.episode_cost_max = np.zeros(self.n_constraints, dtype=np.float32)
        self.episode_length = 0

    def get_lambda(self) -> np.ndarray:
        return self.lam.copy()

    def set_lambda(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.n_constraints,):
            value = np.full(self.n_constraints, float(value.flat[0]), dtype=np.float32)
        self.lam = np.clip(value, 0.0, self.lambda_max)

    def reset(self, **kwargs):
        self.episode_cost = np.zeros(self.n_constraints, dtype=np.float32)
        self.episode_cost_max = np.zeros(self.n_constraints, dtype=np.float32)
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs = np.asarray(
            info.get("constraint_costs", np.zeros(self.n_constraints, dtype=np.float32)),
            dtype=np.float32,
        )
        if costs.shape != (self.n_constraints,):
            costs = np.zeros(self.n_constraints, dtype=np.float32)

        self.episode_cost += costs
        self.episode_cost_max = np.maximum(self.episode_cost_max, costs)
        self.episode_length += 1

        penalty = float(np.dot(self.lam, costs))
        modified_reward = reward - penalty

        info["original_reward"] = reward
        info["constraint_costs"] = costs
        info["lagrangian_penalty"] = penalty
        info["lambda"] = self.lam.copy()

        if (terminated or truncated) and self.lambda_lr > 0.0:
            self._update_lambda()

        if terminated or truncated:
            avg_cost = (
                self.episode_cost / float(self.episode_length)
                if self.episode_length > 0
                else self.episode_cost.copy()
            )
            info["lagrangian/episode_cost_sum"] = self.episode_cost.copy()
            info["lagrangian/episode_cost_avg"] = avg_cost.astype(np.float32)
            info["lagrangian/episode_cost_max"] = self.episode_cost_max.copy()
            info["lagrangian/episode_length"] = int(self.episode_length)

        info["lagrangian/lambda_v"] = float(self.lam[0])
        info["lagrangian/lambda_p"] = float(self.lam[1])
        info["lagrangian/lambda_q"] = float(self.lam[2])
        info["lagrangian/penalty"] = penalty
        info["lagrangian/cost_v"] = float(costs[0])
        info["lagrangian/cost_p"] = float(costs[1])
        info["lagrangian/cost_q"] = float(costs[2])

        return obs, modified_reward, terminated, truncated, info

    def _update_lambda(self):

        if self.normalize and self.episode_length > 0:
            avg_cost = self.episode_cost / self.episode_length
        else:
            avg_cost = self.episode_cost
        
        effective_cost = (
            (1.0 - self.peak_weight) * avg_cost
            + self.peak_weight * self.episode_cost_max
        )

        gradient = effective_cost - self.cost_limit
        self.lam = np.clip(self.lam + self.lambda_lr * gradient, 0.0, self.lambda_max)
        
        print(
            f"Episode ended | Steps: {self.episode_length}\n"
            f"Avg costs:    {avg_cost}  (limit: {self.cost_limit})\n"
            f"Peak costs:   {self.episode_cost_max}\n"
            f"Effective:    {effective_cost}\n"
            f"Gradient:     {gradient}\n"
            f"New lambda:   {self.lam}\n"
        )


class BatteryClipWrapper(gym.Wrapper):

    _FALLBACK_POWER_LIMIT_MW  = 50.0
    _FALLBACK_SOC_MAX_MWH     = 100.0
    _FALLBACK_SOC_MIN_MWH     = 0.0
    _FALLBACK_EFFICIENCY      = 0.9
    _FALLBACK_DELTA_T_H       = 0.25

    def __init__(self, env, clip_penalty_weight: float = 10.0):
        super().__init__(env)
        self._battery_device_id = self._find_storage_device_id(env)
        self.battery_action_idx = self._find_battery_action_idx(env)
        self._clip_penalty_weight = float(clip_penalty_weight)
        # Cached static params (populated on first step)
        self._static_cached = False
        self._delta_t_h = self._FALLBACK_DELTA_T_H
        self._efficiency = self._FALLBACK_EFFICIENCY
        self._power_limit_mw = self._FALLBACK_POWER_LIMIT_MW
        self._soc_max_mwh = self._FALLBACK_SOC_MAX_MWH
        self._soc_min_mwh = self._FALLBACK_SOC_MIN_MWH
        # Cached normalizer reference
        self._normalizer = None
        self._normalizer_cached = False

    @staticmethod
    def _find_storage_device_id(env):
        """Auto-detect the StorageUnit device ID from the simulator."""
        base = env
        while hasattr(base, "env"):
            base = base.env
        if hasattr(base, "simulator"):
            for dev_id, dev in base.simulator.devices.items():
                if type(dev).__name__ == "StorageUnit":
                    return dev_id
        return 6  # fallback

    @staticmethod
    def _find_battery_action_idx(env):
        """Compute the action index for the first StorageUnit P setpoint.

        gym-anm action layout: [P_gen...] [Q_gen...] [P_des...] [Q_des...]
        Battery P starts at index 2 * N_non_slack_gen.
        """
        base = env
        while hasattr(base, "env"):
            base = base.env
        if hasattr(base, "simulator"):
            from gym_anm.simulator.components import Generator, StorageUnit
            n_gen = sum(
                1 for dev in base.simulator.devices.values()
                if isinstance(dev, Generator) and not dev.is_slack
            )
            return 2 * n_gen  # P_des starts after [P_gen * n_gen] + [Q_gen * n_gen]
        return 2  # fallback for BCP (1 generator)

    def _find_normalizer(self):
        if self._normalizer_cached:
            return self._normalizer
        env = self.env
        while env is not None:
            if isinstance(env, NormalizeActionWrapper):
                self._normalizer = env
                self._normalizer_cached = True
                return env
            env = getattr(env, "env", None)
        self._normalizer_cached = True
        return None

    def _cache_static_params(self, base_env):
        """Read static battery parameters once from the simulator and cache them."""
        if self._static_cached:
            return

        using_fallback = []

        delta_t_h = float(getattr(base_env, "delta_t", None) or 0.0)
        if delta_t_h <= 0:
            using_fallback.append(f"delta_t_h={self._FALLBACK_DELTA_T_H} (not found in base_env.delta_t)")
            delta_t_h = self._FALLBACK_DELTA_T_H

        dev_id = self._battery_device_id
        power_limit_mw = None
        soc_max_mwh = None
        soc_min_mwh = None
        efficiency = None

        if hasattr(base_env, "simulator"):
            base_mva = float(getattr(base_env.simulator, "baseMVA", 0.0))
            dev = getattr(base_env.simulator, "devices", {}).get(dev_id, None)

            if dev is not None and base_mva > 0:
                power_limit_mw = float(getattr(dev, "p_max", None) or 0.0) * base_mva
                soc_max_mwh    = float(getattr(dev, "soc_max", None) or 0.0) * base_mva
                soc_min_mwh    = float(getattr(dev, "soc_min", None) or 0.0) * base_mva
                efficiency     = float(getattr(dev, "eff",    None) or 0.0)

                if power_limit_mw <= 0:
                    using_fallback.append(
                        f"power_limit_mw={self._FALLBACK_POWER_LIMIT_MW} "
                        f"(value read from simulator: {power_limit_mw:.3f} <= 0)"
                    )
                    power_limit_mw = self._FALLBACK_POWER_LIMIT_MW
                if soc_max_mwh <= 0:
                    using_fallback.append(
                        f"soc_max_mwh={self._FALLBACK_SOC_MAX_MWH} "
                        f"(value read from simulator: {soc_max_mwh:.3f} <= 0)"
                    )
                    soc_max_mwh = self._FALLBACK_SOC_MAX_MWH
                if efficiency <= 0:
                    using_fallback.append(
                        f"efficiency={self._FALLBACK_EFFICIENCY} "
                        f"(value read from simulator: {efficiency:.3f} <= 0)"
                    )
                    efficiency = self._FALLBACK_EFFICIENCY
                if soc_min_mwh < 0:
                    soc_min_mwh = 0.0
            else:
                using_fallback.append(
                    f"all battery parameters (device {dev_id} "
                    f"not found or baseMVA={base_mva:.1f})"
                )
        else:
            using_fallback.append("all battery parameters (simulator not accessible)")

        self._delta_t_h = delta_t_h
        self._efficiency = efficiency if efficiency is not None else self._FALLBACK_EFFICIENCY
        self._power_limit_mw = power_limit_mw if power_limit_mw is not None else self._FALLBACK_POWER_LIMIT_MW
        self._soc_max_mwh = soc_max_mwh if soc_max_mwh is not None else self._FALLBACK_SOC_MAX_MWH
        self._soc_min_mwh = soc_min_mwh if soc_min_mwh is not None else self._FALLBACK_SOC_MIN_MWH

        if using_fallback:
            warnings.warn(
                f"BatteryClipWrapper: hardcoded values:\n"
                + "\n".join(f"  - {msg}" for msg in using_fallback)
                + "\nIf the network has changed, verify that the fallback values are still correct.",
                stacklevel=3,
            )

        self._static_cached = True

    def _get_current_soc(self, base_env):
        """Read only the dynamic SoC value from the simulator state."""
        if hasattr(base_env, "simulator") and getattr(base_env.simulator, "state", None) is not None:
            return float(
                base_env.simulator.state.get("des_soc", {}).get("MWh", {}).get(self._battery_device_id, 0.0)
            )
        return 0.0

    def step(self, action):
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # Cache static params on first call
        self._cache_static_params(base_env)

        action = np.array(action, dtype=np.float32)
        normalizer = self._find_normalizer()
        raw_action = (
            np.array(normalizer.rescale_action(action), dtype=np.float32)
            if normalizer is not None else action.copy()
        )

        clip_fraction = 0.0  # 0 = no clip, 1 = fully clipped

        if len(raw_action) > self.battery_action_idx:
            current_soc_mwh = self._get_current_soc(base_env)
            intended_mw = float(raw_action[self.battery_action_idx])

            # Max discharge (P > 0): limited by how much energy is currently stored.
            # delta_SoC = -P * delta_t / eff  =>  P_max = (soc - soc_min) * eff / delta_t
            max_discharge_mw = (current_soc_mwh - self._soc_min_mwh) * self._efficiency / max(self._delta_t_h, 1e-8)
            max_discharge_mw = float(np.clip(max_discharge_mw, 0.0, self._power_limit_mw))

            # Max charge (P < 0): limited by remaining capacity.
            # delta_SoC = |P| * delta_t * eff  =>  |P|_max = (soc_max - soc) / (eff * delta_t)
            max_charge_mw = (self._soc_max_mwh - current_soc_mwh) / (self._efficiency * max(self._delta_t_h, 1e-8))
            max_charge_mw = float(np.clip(max_charge_mw, 0.0, self._power_limit_mw))

            raw_action[self.battery_action_idx] = np.clip(
                raw_action[self.battery_action_idx],
                -max_charge_mw,    # negative = charging
                max_discharge_mw,  # positive = discharging
            )

            executed_mw = float(raw_action[self.battery_action_idx])
            if abs(intended_mw) > 1e-6:
                clip_fraction = abs(intended_mw - executed_mw) / abs(intended_mw)

        if normalizer is not None:
            low, high = normalizer.low, normalizer.high
            denom = np.where(np.abs(high - low) < 1e-8, 1.0, high - low)
            clipped_action = np.clip(2 * (raw_action - low) / denom - 1, -1.0, 1.0)
        else:
            clipped_action = raw_action

        obs, reward, terminated, truncated, info = self.env.step(clipped_action)

        # Penalize infeasible battery actions so the agent learns SoC limits
        clip_penalty = self._clip_penalty_weight * clip_fraction
        reward = float(reward) - clip_penalty

        info["battery_clip/fraction"] = clip_fraction
        info["battery_clip/penalty"] = clip_penalty

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class StandardPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, w_v: float = 50.0, w_p: float = 50.0, w_q: float = 50.0):
        super().__init__(env)
        self.weights = np.array([w_v, w_p, w_q], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs = np.asarray(
            info.get("constraint_costs", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        )
        if costs.shape != (3,):
            costs = np.zeros(3, dtype=np.float32)
        penalty = float(np.dot(self.weights, costs))
        modified_reward = reward - penalty
        info["original_reward"] = reward
        info["standard_penalty"] = penalty
        info["standard_penalty/cost_v"] = float(costs[0])
        info["standard_penalty/cost_p"] = float(costs[1])
        info["standard_penalty/cost_q"] = float(costs[2])
        return obs, modified_reward, terminated, truncated, info