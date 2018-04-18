"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""


import gym

from collections import deque


class StackFramesWrapper(gym.Env):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, num_frames):
        self.env = env
        self.num_frames = num_frames
        self.frames = None

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        observation = self.env.reset()
        self.frames = deque([observation] * self.num_frames)
        return self.frames

    def step(self, action):
        new_observation = self.env.step(action)
        self.frames.popleft()
        self.frames.append(new_observation)
        return self.frames

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


class GrayscaleAndResize(gym.Env):
    """Resize observation frames to specified (w,h) and convert to grayscale."""
    pass