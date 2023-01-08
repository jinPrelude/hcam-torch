from cleanrl.pycolab_ballet import ballet_environment

import gym
import numpy as np

class SyncVectorBalletEnv():
  """recieve multiple BalletEnv and return a batches of the obs, rewards, dones, infos"""
  def __init__(self, envs):
    self.envs = envs
    self.num_envs = len(envs)

  def reset(self):
    return [env.reset() for env in self.envs]

  def step(self, actions):
    obs_image, obs_language, rewards, dones, infos = [], [], [], [], []
    for i, env in enumerate(self.envs):
      timestep = env.step(actions[i])
      obs_image.append(timestep[0][0])
      obs_language.append(timestep[0][1])
      rewards.append(timestep[1])
      dones.append(timestep[2])
      infos.append({})
    obs_image = np.stack(obs_image, axis=0)
    obs_language = np.stack(obs_language, axis=0).astype(np.float32)
    return [obs_image, obs_language], rewards, dones, infos

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

envs = SyncVectorBalletEnv([BalletEnv(ballet_environment.simple_builder(level_name='2_delay16')) for _ in range(4)])
timestep = envs.reset()
for _ in range(100):
  action = [x for x in range(4)]
  timestep = envs.step(action)
  print(timestep)