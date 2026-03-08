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