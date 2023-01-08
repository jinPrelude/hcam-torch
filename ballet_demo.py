from cleanrl.pycolab_ballet import ballet_environment

import gym
import numpy as np
class BalletEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        timestep = self.env.step(action)
        obs = timestep.observation
        obs = (np.transpose(obs[0], (-1, 0, 1)), str(obs[1]))
        reward = timestep.reward
        done = timestep.last()
        return obs, reward, done, {}

env = ballet_environment.simple_builder(level_name='2_delay16')
env = BalletEnv(env)
timestep = env.reset()
test = str(timestep.observation[1])
for _ in range(100):
  action = 0
  timestep = env.step(action)
  print(timestep)