"""Add C Axis for image observation from (H, W) to (C, H, W)"""

import numpy as np

import gym
from gym.spaces import Box, Tuple


class NewAxisObservation(gym.ObservationWrapper):
    """Add C Axis for image observation from (H, W) to (C, H, W)"""

    def __init__(self, env: gym.Env):
        """Add C Axis for image observation from (H, W) to (C, H, W).

        Args:
            env (Env): The environment to apply the wrappe
        """
        super().__init__(env)

        assert (
            isinstance(self.observation_space[0], Box)
            and len(self.observation_space[0].shape) == 2
        )

        obs_shape = self.observation_space[0].shape[:2]
        self.observation_space = Tuple(
            (Box(
            low=0, high=255, shape=(1, obs_shape[0], obs_shape[1]), dtype=np.uint8
        ), self.observation_space[1])
        )

    def observation(self, observation):
        image_obs, lang_obs = observation
        return (image_obs[np.newaxis, :], lang_obs)