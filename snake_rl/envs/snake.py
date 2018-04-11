"""
Implementation of Snake game.

"""


import os
import gym
import pygame
import logging
import numpy as np

from gym import spaces
from gym.utils import seeding

from snake_rl.utils import Vec


logger = logging.getLogger(os.path.basename(__file__))


class Action:
    """Possible actions in the environment."""

    all_actions = range(5)
    up, right, down, left, noop = all_actions

    movement = {
        up: (-1, 0),
        right: (0, 1),
        down: (1, 0),
        left: (0, -1),
        noop: (0, 0),
    }

    @staticmethod
    def delta(action):
        return Vec(*Action.movement[action])


class Entity:
    """Basically, anything within the game world."""

    def _color(self):
        """Default color, should be overridden."""
        return 255, 0, 255

    def draw(self, game, surface, pos, scale):
        game.draw_tile(surface, pos, self._color(), scale)


class Terrain(Entity):
    """An abstract unit of terrain."""
    pass


class Ground(Terrain):
    """Default terrain."""

    def _color(self):
        return 39, 40, 34


class GameObject(Entity):
    """Any object in the game that the snake can interact with."""

    def interact(self, game):
        """Returns reward. Called when the head of the snake is in contact with the object."""
        return 0

    def _color(self):
        """Default color, should be overridden."""
        return 255, 0, 255


class Apple(GameObject):
    """Collectable bonus."""

    def interact(self, game):
        """Returns reward."""
        return 0

    def _color(self):
        return 255, 191, 0


class SnakeBody(GameObject):
    """Part of the snake (one tile)."""

    def interact(self, game):
        """Returns reward."""
        return 0

    def _color(self):
        return 255, 255, 255


class GameMode:
    """All the gameplay settings in one place."""

    def __init__(self):
        self.world_size = 10


class Snake(gym.Env):
    """Implementation of the Snake learning environment."""

    reward_unit = 0.1
    max_reward_abs = 100 * reward_unit

    def __init__(self, mode=None):
        pygame.init()

        # use default settings if play mode is None
        self.mode = GameMode() if mode is None else mode
        world_size = self.mode.world_size

        # generate random seed
        self.rng = None
        self.seed()

        # gym.Env variables
        self.action_space = spaces.Discrete(len(Action.all_actions))
        self.observation_space = spaces.Box(0.0, 1.0, shape=[world_size, world_size], dtype=np.float32)
        self.reward_range = (-self.max_reward_abs, self.max_reward_abs)

        # variables for holding the game state
        self.world = None

        self.clock = pygame.time.Clock()

    def seed(self, seed=None):
        """
        Initialize RNG, used for consistent reproducibility.
        Implementation of the gym.Env abstract method.
        """
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Generate the new game instance, overrides gym.Env method."""
        dim = self.mode.world_size

        ground = Ground()
        self.world = np.full((dim, dim), ground, dtype=Entity)

    def step(self, action):
        pass

    def render(self, mode='human'):
        """Implementation of the gym.Env abstract method."""
        pass

    def close(self):
        pass

