"""
Algo utils tests.

"""

import gym
import unittest
import numpy as np

from snake_rl import envs

from snake_rl.algorithms.utils.multi_env import MultiEnv

TEST_ENV = envs.SNAKE_SIMPLE_LATEST


class MultiEnvTest(unittest.TestCase):
    def test_multi_env(self):
        def make_env_func():
            return gym.make(TEST_ENV)

        num_envs = 8
        multi_env = MultiEnv(num_envs=num_envs, make_env_func=make_env_func)
        obs = multi_env.initial_observations()

        self.assertEqual(len(obs), num_envs)

        num_different = 0
        for i in range(1, len(obs)):
            if not np.array_equal(obs[i - 1], obs[i]):
                num_different += 1

        # By pure chance some of the environments might be identical even with different seeds, but definitely not
        # all of them!
        self.assertGreater(num_different, len(obs) // 2)

        for i in range(20):
            obs, rewards, dones = multi_env.step([0] * num_envs)
            self.assertEqual(len(obs), num_envs)
            self.assertEqual(len(rewards), num_envs)
            self.assertEqual(len(dones), num_envs)

        multi_env.close()
