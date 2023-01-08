from cleanrl.pycolab_ballet import ballet_environment

import gym
import numpy as np
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

env = ballet_environment.simple_builder(level_name='2_delay16')
env = BalletEnv(env)
timestep = env.reset()
test = str(timestep.observation[1])
for _ in range(300):
  action = 0
  timestep = env.step(action)
  print(timestep[0][1])