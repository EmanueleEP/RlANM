"""
This file contains a series of wrapper classes used to wrap
gym environments.
"""
import gymnasium as gym
import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """
    This class normalizes the action space of the environment to [-1, 1]^N.

    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    # def reset(self):
    #     """
    #     Reset the environment
    #     """
    #     # Reset the counter
    #     return self.env.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment with optional seed and options."""
        return self.env.reset(seed=seed, options=options)

    # def step(self, action):
    #     """
    #     :param action: ([float] or int) Action taken by the agent
    #     :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    #     """
    #     # Rescale action from [-1, 1] to original [low, high] interval
    #     rescaled_action = self.rescale_action(action)
    #     obs, reward, done, info = self.env.step(rescaled_action)
    #     return obs, reward, done, info

    def step(self, action):
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info  = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info



class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, seed=None, options=None):
        self._elapsed_steps = 0  # reset the counter here
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info