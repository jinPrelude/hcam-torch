"""gray_scale_observation for balletenv.
original code
    : https://github.com/openai/gym/blob/master/gym/wrappers/gray_scale_observation.py
"""
import os

import numpy as np
import cv2
import imageio
import pygifsicle
import gym
from gym import logger
from gym.spaces import Box, Tuple, Discrete


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.
    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...
    Args:
        episode_id: The episode number
    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class RecordVideo(gym.Wrapper):

    def __init__(self, env: gym.Env, path: str, fps: int = 15):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)

        assert (
            isinstance(self.observation_space[0], Box)
            and len(self.observation_space[0].shape) == 3
        )
        assert self.observation_space[0].shape[-1] == 3, "observation space must be 3D. Did you use GrayScaleObservation first?"
        self.episode_id = 0
        self.path = path
        self.fps = fps
        self.video_lst = []
        self.recording = False
        self.done = False

        self.path = os.path.abspath(path)
        # Create output folder if needed
        if os.path.isdir(self.path):
            logger.warn(
                f"Overwriting existing videos at {self.path} folder "
                f"(try specifying a different `path` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.path, exist_ok=True)

        self.episode_trigger = capped_cubic_video_schedule

    def _render_frame(self, obs: np.array, lang: str=None):
        """create black canvas which the size is (99, 200, 3),
        draw obs on the left side and write lang on the right side for rendering cv2"""
        canvas = np.zeros((99, 200, 3), dtype=np.uint8)
        canvas[:, :99, :] = obs
        if lang:
            cv2.putText(canvas, lang, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def start_video_recording(self, observation):
        # self.close_video_recorder()
        frame = self._render_frame(observation[0])
        self.video_lst.append(frame)
        self.recording = True
    
    def capture_video(self, observation):
        frame = self._render_frame(observation[0])
        self.video_lst.append(frame)

    def close_video_recorder(self):
        self.recording = False

        video_name = f"rl-video-episode-{self.episode_id}.gif"
        base_path = os.path.join(self.path, video_name)

        imageio.mimsave(base_path, self.video_lst, fps=self.fps)
        pygifsicle.optimize(base_path)
        self.video_lst = []


    def _video_enabled(self):
        return self.episode_trigger(self.episode_id)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.done = False
        if self._video_enabled():
            self.start_video_recording(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)

        if not self.done:
            if done:
                self.episode_id += 1
                self.done = True

        if self.recording:
            frame = self._render_frame(observation[0], info['instruction_string'])
            self.video_lst.append(frame)

            if done:
                self.close_video_recorder()

        return observation, reward, done, info