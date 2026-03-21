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
    """Normalizes the action space to [-1, 1]^N."""

    def __init__(self, env):
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box), \
            "This wrapper only works with continuous action space (spaces.Box)"
        self.low, self.high = action_space.low, action_space.high
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """Rescale the action from [-1, 1] to [low, high]."""
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    """Enforces a maximum episode length."""

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
    """Early-terminate the episode when the power flow does not converge."""

    def __init__(self, env: gym.Env, penalty: float = 100.0):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pfe_converged = info.get("pfe_converged", True)
        if pfe_converged is False:
            reward = float(reward) - self.penalty
            terminated = True
            info["pfe_terminated"] = True
            info["pfe_penalty"] = float(self.penalty)
        return obs, reward, terminated, truncated, info


class LagrangianWrapper(gym.Wrapper):
    """Applies Lagrangian penalty on constraint violations and updates λ via dual ascent."""

    def __init__(
        self,
        env: gym.Env,
        v_tolerance: float = 0.05,
        pq_limit: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 1e-2,
        normalize_by_episode_length: bool = True,
        lambda_max: float = 500.0,
    ):
        super().__init__(env)
        base_env = env.unwrapped
        self.n_buses = base_env.simulator.N_bus
        self.v_tol = v_tolerance
        self.lambda_lr = lambda_lr
        self.normalize = normalize_by_episode_length
        self.lambda_max = lambda_max
        self.n_constraints = 3
        self.cost_limit = np.array([v_tolerance, pq_limit, pq_limit], dtype=np.float32)

        lambda_init_arr = np.atleast_1d(np.array(lambda_init, dtype=np.float32))
        if lambda_init_arr.shape == (self.n_constraints,):
            self.lam = lambda_init_arr
        else:
            self.lam = np.full(self.n_constraints, float(lambda_init_arr[0]), dtype=np.float32)

        self.episode_cost = np.zeros(self.n_constraints, dtype=np.float32)
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
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs = info.get("constraint_costs", np.zeros(self.n_constraints, dtype=np.float32))
        costs = np.asarray(costs, dtype=np.float32)
        if costs.shape != (self.n_constraints,):
            costs = np.zeros(self.n_constraints, dtype=np.float32)

        self.episode_cost += costs
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
            avg_cost = (self.episode_cost / float(self.episode_length)
                        if self.episode_length > 0 else self.episode_cost.copy())
            info["lagrangian/episode_cost_sum"] = self.episode_cost.copy()
            info["lagrangian/episode_cost_avg"] = avg_cost.astype(np.float32)
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
        gradient = avg_cost - self.cost_limit
        self.lam = np.clip(self.lam + self.lambda_lr * gradient, 0.0, self.lambda_max)
        print(
            f"Episode ended | Steps: {self.episode_length}\n"
            f"Avg costs:  {avg_cost}  (limit: {self.cost_limit})\n"
            f"Gradient:   {gradient}\n"
            f"New lambda: {self.lam}\n"
        )


class BatteryClipWrapper(gym.Wrapper):
    """
    Clips battery actions to respect physical power limits and SOC constraints.
    """

    def __init__(self, env):
        super().__init__(env)
        self.battery_action_idx = 2
        self._default_power_limit_mw = 50.0
        self._default_soc_max_mwh = 100.0
        self._default_soc_min_mwh = 0.0
        self._default_efficiency = 0.9
        self._default_delta_t_h = 0.25

    def _find_normalizer(self):
        """Traverse the wrapper chain to find NormalizeActionWrapper."""
        env = self.env
        while env is not None:
            if isinstance(env, NormalizeActionWrapper):
                return env
            env = getattr(env, 'env', None)
        return None

    def step(self, action):
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        action = np.array(action, dtype=np.float32)

        normalizer = self._find_normalizer()
        if normalizer is not None:
            raw_action = np.array(normalizer.rescale_action(action), dtype=np.float32)
        else:
            raw_action = action.copy()

        battery_device_id = 3
        delta_t_h = float(getattr(base_env, "delta_t", self._default_delta_t_h))
        efficiency = self._default_efficiency
        power_limit_mw = self._default_power_limit_mw
        soc_max_mwh = self._default_soc_max_mwh
        soc_min_mwh = self._default_soc_min_mwh
        current_soc_mwh = 0.0

        if hasattr(base_env, "simulator") and getattr(base_env.simulator, "state", None) is not None:
            full_state = base_env.simulator.state
            current_soc_mwh = float(
                full_state.get("des_soc", {}).get("MWh", {}).get(battery_device_id, current_soc_mwh)
            )
            base_mva = float(getattr(base_env.simulator, "baseMVA", 0.0)) or None
            dev = getattr(base_env.simulator, "devices", {}).get(battery_device_id, None)
            if dev is not None and base_mva is not None:
                power_limit_mw = float(getattr(dev, "p_max", power_limit_mw / base_mva)) * base_mva
                soc_max_mwh = float(getattr(dev, "soc_max", soc_max_mwh / base_mva)) * base_mva
                soc_min_mwh = float(getattr(dev, "soc_min", soc_min_mwh / base_mva)) * base_mva
                efficiency = float(getattr(dev, "eff", efficiency))

        original_battery_p = None
        if len(raw_action) > self.battery_action_idx:
            battery_action = raw_action[self.battery_action_idx]
            original_battery_p = float(battery_action)
            clipped_battery_action = np.clip(battery_action, -power_limit_mw, power_limit_mw)

            if clipped_battery_action > 0:  # Discharge
                usable_energy_mwh = max(0.0, current_soc_mwh - soc_min_mwh)
                if delta_t_h <= 0:
                    max_discharge_power = 0.0
                else:
                    # SoC_new = SoC_old - P*dt/eff  →  P_max = usable*eff/dt
                    max_discharge_power = usable_energy_mwh * max(efficiency, 1e-6) / delta_t_h
                max_discharge_power = min(max_discharge_power, power_limit_mw)
                clipped_battery_action = min(clipped_battery_action, max_discharge_power)

            elif clipped_battery_action < 0:  # Charge
                max_energy_add = max(0.0, soc_max_mwh - current_soc_mwh)
                if delta_t_h <= 0:
                    max_charge_power = 0.0
                else:
                    # SoC_new = SoC_old + |P|*dt*eff  →  P_max = capacity/(dt*eff)
                    max_charge_power = max_energy_add / (delta_t_h * max(efficiency, 1e-6))
                max_charge_power = min(max_charge_power, power_limit_mw)
                clipped_battery_action = max(clipped_battery_action, -max_charge_power)

            raw_action[self.battery_action_idx] = clipped_battery_action

        # Azzera Q se la batteria è fisicamente ferma per vincoli SoC
        battery_q_idx = self.battery_action_idx + 1
        if len(raw_action) > battery_q_idx and original_battery_p is not None:
            clipped_p = raw_action[self.battery_action_idx]
            if abs(original_battery_p) > 1e-3 and abs(clipped_p) < 0.05 * abs(original_battery_p):
                raw_action[battery_q_idx] = 0.0

        if normalizer is not None:
            low = normalizer.low
            high = normalizer.high
            denom = high - low
            denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
            clipped_action = 2 * (raw_action - low) / denom - 1
            clipped_action = np.clip(clipped_action, -1.0, 1.0)
        else:
            clipped_action = raw_action

        obs, reward, terminated, truncated, info = self.env.step(clipped_action)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class StandardPenaltyWrapper(gym.Wrapper):
    """Applies fixed constraint penalties directly to the reward (non-Lagrangian training)."""

    def __init__(self, env: gym.Env, w_v: float = 50.0, w_p: float = 50.0, w_q: float = 50.0):
        super().__init__(env)
        self.weights = np.array([w_v, w_p, w_q], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs = info.get("constraint_costs", np.zeros(3, dtype=np.float32))
        costs = np.asarray(costs, dtype=np.float32)
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