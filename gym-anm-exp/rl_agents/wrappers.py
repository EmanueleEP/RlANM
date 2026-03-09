"""
This file contains a series of wrapper classes used to wrap
gym environments.
"""
import gymnasium as gym
import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    Normalizes the action space of the environment to [-1, 1]^N.

    :param env: (gym.Env) Gym environment that will be wrapped
    """

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
        """
        Rescale the action from [-1, 1] to [low, high].
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    """
    Enforces a maximum episode length by setting truncated=True
    after max_episode_steps steps.
    """

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


class LagrangianWrapper(gym.Wrapper):
    
    # v_tolerance:              max voltage deviation from 1.0 p.u. (default ±5%).
    # cost_limit:               Target average violation per step (default 0.0 = no violation).
    # lambda_init:              Initial value of the Lagrange multiplier.
    # lambda_lr:                Dual learning rate for lambda updates.
    # normalize_by_episode_length: If True, use per-step average cost for the update.
    # lambda_max:               Maximum allowed lambda value (clips to [0, lambda_max]).
    

    def __init__(
        self,
        env: gym.Env,
        v_tolerance: float = 0.05,
        pq_limit: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 5e-5,
        normalize_by_episode_length: bool = True,
        lambda_max: float = 100.0,
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

    def get_lambda(self) -> float:
        """Return the current lambda value (useful for checkpointing)."""
        return self.lam

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

        # Defensive get in case constraint_costs is missing (e.g. on first step or env variants)
        costs = info.get("constraint_costs", np.zeros(self.n_constraints, dtype=np.float32))
        
        # Validate constraint costs have correct shape
        costs = np.asarray(costs, dtype=np.float32)
        if costs.shape != (self.n_constraints,):
            costs = np.zeros(self.n_constraints, dtype=np.float32)

        # accumulate episode statistics
        self.episode_cost += costs
        self.episode_length += 1

        # lagrangian penalty
        penalty = float(np.dot(self.lam, costs))
        modified_reward = reward - penalty

        # logging
        info["original_reward"] = reward
        info["constraint_costs"] = costs
        info["lagrangian_penalty"] = penalty
        info["lambda"] = self.lam.copy()

        # update lambda
        if (terminated or truncated) and self.lambda_lr > 0.0:
            self._update_lambda()
        
        info["lagrangian/lambda_v"] = float(self.lam[0])
        info["lagrangian/lambda_p"] = float(self.lam[1])
        info["lagrangian/lambda_q"] = float(self.lam[2])
        info["lagrangian/penalty"] = penalty
        info["lagrangian/cost_v"] = float(costs[0])
        info["lagrangian/cost_p"] = float(costs[1])
        info["lagrangian/cost_q"] = float(costs[2])

        return obs, modified_reward, terminated, truncated, info

    def _update_lambda(self):
        """Gradient ascent step on the dual variable."""
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
    Clips battery actions to respect physical power limits AND SOC constraints while allowing
    the agent to request any action in the normalized action space.
    
    This wrapper ensures that battery charging/discharging actions are
    within the device's physical limits (±50 MW) and respect SOC constraints.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Battery P setpoint is at action[2] in the BCP environment
        # Action mapping: [RenewableGen_P, RenewableGen_Q, StorageUnit_P, StorageUnit_Q]
        self.battery_action_idx = 2
        self.battery_power_limit = 50.0  # MW (from network.py: PMAX = 50, PMIN = -50)
        self.battery_soc_max = 100.0  # MWh (from network.py: SOC_MAX = 100)
        self.battery_efficiency = 0.9  # from network.py: EFF = 0.9
        self.delta_t = 1.0  # 1 hour timestep (default in gym-anm)
        
    def step(self, action):
        # Get the base environment to access the simulator
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        # Check if this is a battery action (action[2] affects device 3)
        if len(action) > 2:
            # Rescale action to get the raw values
            if hasattr(self.env, 'rescale_action'):
                raw_action = self.env.rescale_action(action)
            else:
                raw_action = action
                
            # Get current battery SOC if available
            current_soc = 0.0
            if hasattr(base_env, 'simulator'):
                full_state = base_env.simulator.state
                des_soc = list(full_state["des_soc"]["MWh"].values())
                if des_soc:
                    current_soc = des_soc[0]  # First (and only) storage device
                
            # Clip battery power to physical limits AND SOC constraints
            if len(raw_action) > self.battery_action_idx:
                battery_action = raw_action[self.battery_action_idx]
                
                # Apply power limits first
                clipped_battery_action = np.clip(battery_action, -self.battery_power_limit, self.battery_power_limit)
                
                # Apply SOC constraints
                if clipped_battery_action > 0:  # Charging
                    # Check if battery is full
                    max_charge_power = (self.battery_soc_max - current_soc) / self.delta_t
                    max_charge_power = min(max_charge_power, self.battery_power_limit)
                    clipped_battery_action = min(clipped_battery_action, max_charge_power)
                    
                elif clipped_battery_action < 0:  # Discharging
                    # Check if battery is empty - allow some minimum SOC for discharge
                    if current_soc <= 0.1:  # Allow small discharge even when nearly empty
                        max_discharge_power = 0.1 / (self.delta_t * self.battery_efficiency)
                    else:
                        max_discharge_power = current_soc / (self.delta_t * self.battery_efficiency)
                    max_discharge_power = min(max_discharge_power, self.battery_power_limit)
                    clipped_battery_action = max(clipped_battery_action, -max_discharge_power)
                
                raw_action[self.battery_action_idx] = clipped_battery_action
                
                # Debug output for monitoring - always show for first few steps
                if abs(battery_action - clipped_battery_action) > 0.01 or current_soc < 1.0:
                    print(f"Battery: {battery_action:.3f} -> {clipped_battery_action:.3f} MW (SOC: {current_soc:.3f}/{self.battery_soc_max:.1f} MWh)")
                
                # If we have a normalized wrapper, we need to use the clipped action
                if hasattr(self.env, 'rescale_action'):
                    # Step with the clipped raw action
                    obs, reward, terminated, truncated, info = self.env.step(raw_action)
                else:
                    # Step with the clipped action directly
                    obs, reward, terminated, truncated, info = self.env.step(raw_action)
            else:
                obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)