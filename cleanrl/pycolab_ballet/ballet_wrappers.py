from multiprocessing import Process

import numpy as np
import random
import gym

class SyncVectorBalletEnv():
    """recieve multiple BalletEnv and return a batches of the obs, rewards, dones, infos"""
    def __init__(self, envs, process_num):
        self.envs = envs
        self.num_envs = len(envs)
        assert self.num_envs % process_num == 0

        self.episode_returns = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs

    def reset(self):
        obs = [env.reset() for env in self.envs]
        obs_image = np.stack([o[0] for o in obs], axis=0)
        obs_language = np.stack([o[1] for o in obs], axis=0).astype(np.float32)
        return (obs_image, obs_language)

    def step(self, actions):
        obs_images, obs_languages, rewards, dones, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            timestep = env.step(actions[i])
            obs_image = timestep[0][0]
            obs_language = timestep[0][1]
            reward = timestep[1]
            done = timestep[2]
            
            self.episode_returns[i] += timestep[1]
            self.episode_lengths[i] += 1
            if done:
                infos.append({"episode": {"r": self.episode_returns[i], "l": self.episode_lengths[i]}})
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
                (obs_image, obs_language) = env.reset()
            else:
                infos.append({})

            obs_images.append(obs_image)
            obs_languages.append(obs_language)
            rewards.append(reward)
            dones.append(done)

        obs_images = np.stack(obs_images, axis=0)
        obs_languages = np.stack(obs_languages, axis=0).astype(np.float32)

        return [obs_images, obs_languages], rewards, dones, infos

class BalletEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.language_dict = {
            "watch": 0,
            "circle_cw": 1,
            "circle_ccw": 2,
            "up_and_down": 3,
            "left_and_right": 4,
            "diagonal_uldr": 5,
            "diagonal_urdl": 6,
            "plus_cw": 7,
            "plus_ccw": 8,
            "times_cw": 9,
            "times_ccw": 10,
            "zee": 11,
            "chevron_down": 12,
            "chevron_up": 13
        }
    
    def reset(self):
        timestep = self.env.reset()
        obs = timestep.observation
        one_hot = self.language_dict[str(obs[1])]
        language_one_hot_vector = np.zeros(14)
        language_one_hot_vector[one_hot] = 1
        obs = (np.transpose(obs[0], (-1, 0, 1)), language_one_hot_vector)
        return obs

    def step(self, action):
        timestep = self.env.step(action)
        obs = timestep.observation
        one_hot = self.language_dict[str(obs[1])]
        language_one_hot_vector = np.zeros(14)
        language_one_hot_vector[one_hot] = 1
        obs = (np.transpose(obs[0], (-1, 0, 1)), language_one_hot_vector)
        reward = timestep.reward
        done = timestep.last()
        return obs, reward, done, {}