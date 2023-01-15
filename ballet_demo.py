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

env = ballet_environment.simple_builder(level_name='2_delay16')
env = BalletEnv(env)
timestep = env.reset()
test = str(timestep.observation[1])
for _ in range(300):
    action = 0
    timestep = env.step(action)
    print(timestep[0][1])
    '''
    obs = timestep[0][0][1]*255
    obs = obs.astype(np.uint8)
    new_obs = []
    for i in (obs[0], obs[10], obs[20], obs[30], obs[40], obs[50], obs[60], obs[70], obs[80], obs[90]):
        new_obs.append([i[0], i[10], i[20], i[30], i[40], i[50], i[60], i[70], i[80], i[90]])
    new_obs = np.array(new_obs)
    print(new_obs)
    print(timestep[3]["language"])
    input()
    '''