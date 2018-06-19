"""
A2C module tests.

"""

import shutil
import unittest
import numpy as np

from snake_rl import envs

from snake_rl.algorithms.baselines import a2c

from snake_rl.utils.misc import experiment_dir
from snake_rl.utils.logs import get_test_logger


logger = get_test_logger()

TEST_ENV = envs.SNAKE_SIMPLE_LATEST


class A2CTest(unittest.TestCase):
    @staticmethod
    def test_discounted_reward():
        gamma = 0.9
        value = 100.0
        rewards = [1, 2, 3, 4, 5]
        dones = [
            [True, False, False, True, False],
            [True, True, True, True, True],
            [False, False, False, False, True],
        ]
        expected = [
            [1, 7.94, 6.6, 4, 95],
            rewards,
            [11.4265, 11.585, 10.65, 8.5, 5],
        ]

        for done_mask, expected_result in zip(dones, expected):
            calculated = list(a2c.AgentA2C._calc_discounted_rewards(gamma, rewards, done_mask, value))
            np.testing.assert_array_almost_equal(calculated, expected_result)

    def test_train_and_run(self):
        experiment_name = 'a2c_test'
        a2c_params = a2c.AgentA2C.Params(experiment_name)
        a2c_params.train_for_steps = 10
        a2c_params.save_every = a2c_params.train_for_steps - 1
        self.assertEqual(a2c.train_a2c.train(a2c_params, TEST_ENV), 0)
        self.assertEqual(a2c.enjoy_a2c.enjoy(experiment_name, TEST_ENV, max_num_episodes=1, fps=500), 0)
        shutil.rmtree(experiment_dir(experiment_name))
