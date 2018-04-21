"""
Some tests for the Snake environment and utilities.

"""

import gym
import unittest
import numpy as np

from snake_rl import envs

TEST_ENV = envs.SNAKE_SIMPLE_LATEST


class SnakeEnvTests(unittest.TestCase):
    def test_reproducibility_rng_seed(self):
        def generate_env():
            env = gym.make(TEST_ENV)
            env.seed(0)
            return env

        env1, env2 = generate_env(), generate_env()
        for i in range(100):
            # make sure generated worlds are the same
            self.assertTrue(np.array_equal(env1.reset(), env2.reset()))

    def test_render_rgb_array(self):
        env = gym.make(TEST_ENV)
        env.reset()
        array = env.render(mode='rgb_array')
        self.assertIsNotNone(array)


class ResizeGrayScaleTests(unittest.TestCase):
    def test_resize_grayscale(self):
        env = gym.make(TEST_ENV)
        sz = 32
        env = envs.ResizeAndGrayscaleWrapper(env, sz, sz)
        env.reset()
        obs, *_ = env.step(envs.Action.noop)
        self.assertEqual(obs.shape, (sz, sz))
        self.assertEqual(obs.dtype, np.float32)
        for pixel in obs.flatten():
            self.assertTrue(0.0 <= pixel <= 1.0)


class StackFramesTests(unittest.TestCase):
    def test_stack_frames(self):
        env = gym.make(TEST_ENV)
        env = envs.ResizeAndGrayscaleWrapper(env, 64, 64)
        num_frames = 3
        env = envs.StackFramesWrapper(env, num_frames)
        obs = env.reset()
        self.assertEqual(obs.shape[-1], num_frames)
        for i in range(1, obs.shape[-1]):
            self.assertTrue(np.array_equal(obs[:, :, i], obs[:, :, i - 1]))

        for _ in range(num_frames - 1):
            obs, *_ = env.step(envs.Action.noop)

        for i in range(1, obs.shape[-1]):
            self.assertFalse(np.array_equal(obs[:, :, i], obs[:, :, i - 1]))
