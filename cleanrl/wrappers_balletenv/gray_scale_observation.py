"""gray_scale_observation for balletenv.
original code
    : https://github.com/openai/gym/blob/master/gym/wrappers/gray_scale_observation.py
"""
import numpy as np

import gym
from gym.spaces import Box, Tuple, Discrete


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert the image observation from RGB to gray scale.

    Example:
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'))
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'), keep_dim=True)
        >>> env.observation_space
        Box(0, 255, (96, 96, 1), uint8)
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim

        assert (
            isinstance(self.observation_space[0], Box)
            and len(self.observation_space[0].shape) == 3
            and self.observation_space[0].shape[-1] == 3
        )

        obs_shape = self.observation_space[0].shape[:2]
        if self.keep_dim:
            self.observation_space = Tuple(
                (Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            ), self.observation_space[1])
            )
        else:
            self.observation_space = Tuple(
                (Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            ), self.observation_space[1])
            )

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations
        """
        import cv2
        image_obs, lang_obs = observation
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            image_obs = np.expand_dims(image_obs, -1)
        return (image_obs, lang_obs)