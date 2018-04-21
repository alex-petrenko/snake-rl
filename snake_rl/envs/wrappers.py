"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""


import cv2
import gym
import numpy as np

from gym import spaces
from collections import deque

from snake_rl.utils.numpy_utils import numpy_all_the_way


class EnvWrapper(gym.Env):
    """
    Generic wrapper, just forwards all calls to the original env object.
    """

    def __init__(self, env):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range

        self.clock = env.clock

        self.process_events = env.process_events
        self.should_quit = env.should_quit

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


class StackFramesWrapper(EnvWrapper):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, num_frames):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 2:
            raise Exception('Stack frames works with 2D single channel images')
        self.num_frames = num_frames
        self.frames = None

        new_obs_space_shape = env.observation_space.shape + (num_frames, )
        self.observation_space = spaces.Box(0.0, 1.0, shape=new_obs_space_shape, dtype=np.float32)

    def _frames_as_numpy(self):
        return np.transpose(numpy_all_the_way(self.frames), axes=[1, 2, 0])

    def reset(self):
        observation = self.env.reset()
        self.frames = deque([observation] * self.num_frames)
        return self._frames_as_numpy()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self.frames.popleft()
        self.frames.append(new_observation)
        return self._frames_as_numpy(), reward, done, info


class ResizeAndGrayscaleWrapper(EnvWrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h):
        super(ResizeAndGrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, shape=[w, h], dtype=np.float32)
        self.w = w
        self.h = h

    def _observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info


def wrap_env(env, sz=32, num_frames=3):
    env = ResizeAndGrayscaleWrapper(env, sz, sz)
    env = StackFramesWrapper(env, num_frames)
    return env
