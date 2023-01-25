import pytest
import numpy as np
from cleanrl.pycolab_ballet import ballet_environment
from cleanrl.pycolab_ballet.ballet_wrappers import SyncVectorBalletEnv, BalletEnv

def test_SyncVectorBalletEnv():
    fake_envs = [BalletEnv(ballet_environment.simple_builder(level_name='2_delay2')) for i in range(8)]
    process_num = 4
    sync_vector_env = SyncVectorBalletEnv(fake_envs, process_num)

    obs = sync_vector_env.reset()
    assert obs[0].shape == (8, 3, 99, 99)
    assert obs[1].shape == (8, 14)

    actions = np.random.randint(0, 2, (8,))
    obs, rewards, dones, infos = sync_vector_env.step(actions)
    assert obs[0].shape == (8, 3, 99, 99)
    assert obs[1].shape == (8, 14)
    assert rewards == [0.0] * 8
    assert dones == [False] * 8
    assert len(infos) == 8
    assert all([info == {} for info in infos])
    assert all([sync_vector_env.episode_lengths[i] == 1 for i in range(len(sync_vector_env.episode_lengths))])
