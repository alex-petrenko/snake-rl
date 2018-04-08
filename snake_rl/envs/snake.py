"""
Implementation of Snake game.

"""


import gym

from snake_rl.utils.rng import np_random


class Snake(gym.Env):
    """Implementation of the Snake learning environment."""
    def __init__(self):
        # generate random seed
        self.rng = None
        self.seed()

    def seed(self, seed=None):
        """
        Initialize RNG, used for consistent reproducibility.
        Implementation of the gym.Env abstract method.
        """
        self.rng, seed = np_random(seed)
        return [seed]

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        """Implementation of the gym.Env abstract method."""
        pass

    def close(self):
        pass

