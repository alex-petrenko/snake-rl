"""
Implementation of the Gym-compatible Snake game.

"""


import os
import gym
import pygame
import logging
import numpy as np

from collections import deque

from gym import spaces
from gym.utils import seeding

from snake_rl.utils import Vec

from snake_rl.envs.pygame_utils import get_events


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


class GameTile:
    """Basically, anything within the game world."""

    types = range(3)
    ground, snake, apple = types

    @classmethod
    def color(cls, tile):
        return {
            cls.ground: (0, 0, 0),
            cls.snake: (255, 255, 255),
            cls.apple: (255, 0, 0),
        }[tile]

    @staticmethod
    def snake_head_color():
        return 255, 255, 153


class GameMode:
    """All the gameplay settings in one place."""

    def __init__(self):
        self.world_size = 6

    @staticmethod
    def snake(size):
        mode = GameMode()
        mode.world_size = size
        return mode


class Snake(gym.Env):
    """Implementation of the Snake learning environment."""

    render_modes = ['human', 'rgb_array']
    resolution = {'human': 800, 'rgb_array': 50}  # width and height of the game screen as rendered in pixels
    reward_unit = 0.1
    max_reward_abs = 100 * reward_unit

    def __init__(self, mode=None):
        pygame.init()

        # use default settings if play mode is None
        self.mode = GameMode() if mode is None else mode

        # generate random seed
        self.rng = None
        self.seed()

        # gym.Env variables
        self.action_space = spaces.Discrete(len(Action.all_actions))
        self.observation_space = None  # defined in a wrapper (use with a wrapper)
        self.reward_range = (-self.max_reward_abs, self.max_reward_abs)

        # variables for holding the game state
        self.quit = self.over = False
        self.world = None
        self.snake = None
        self.speed = None
        self.num_food = self.steps = self.last_meal = 0

        # stuff related to game rendering
        self.screen = None
        self.tile_sz = {mode: self._tile_sz_for_resolution(res) for mode, res in self.resolution.items()}
        self.rendering_surface = {mode: None for mode in self.render_modes}

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
        self.over = False
        self.steps = self.num_food = self.last_meal = 0

        dim = self.mode.world_size

        self.world = np.full((dim, dim), GameTile.ground, dtype=int)

        initial_snake_length = 2
        self.snake = deque()
        head_position = Vec(dim // 2 - initial_snake_length // 2 + 1, dim // 2)
        for i in range(initial_snake_length):
            pos = Vec(head_position.i + i, head_position.j)
            self.world[pos.ij] = GameTile.snake
            self.snake.append(pos)

        self._spawn_food()  # generate a first piece of food

        self.speed = Action.delta(Action.up)

        return self._get_state()

    def _num_vacant_tiles(self):
        return self.mode.world_size ** 2 - len(self.snake)

    def _spawn_food(self):
        food_idx = self.rng.choice(self._num_vacant_tiles())
        count_vacant = 0
        for (i, j), tile in np.ndenumerate(self.world):
            if tile == GameTile.ground:
                if food_idx == count_vacant:
                    self.world[i, j] = GameTile.apple
                    self.num_food += 1
                    break
                count_vacant += 1

    def should_quit(self):
        """
        :return: True if the used decided to quit the game altogether (e.g. closed the window).
        """
        return self.quit

    def process_events(self):
        """Process key presses, mainly for human play mode."""
        event_types = [e.type for e in get_events()]
        if pygame.QUIT in event_types:
            self.over = self.quit = True
            return Action.noop

        selected_action = Action.noop
        a_map = {
            pygame.K_UP: Action.up,
            pygame.K_DOWN: Action.down,
            pygame.K_LEFT: Action.left,
            pygame.K_RIGHT: Action.right,
        }
        pressed = pygame.key.get_pressed()
        for key, action in a_map.items():
            if pressed[key]:
                selected_action = action
                break

        return selected_action

    def _is_oob(self, pos):
        """Return true if given position is out of world bounds."""
        return pos.i < 0 or pos.i >= self.mode.world_size or pos.j < 0 or pos.j >= self.mode.world_size

    def _hunger(self):
        """Steps since last meal."""
        return self.steps - self.last_meal

    def _stamina(self):
        """How long snake can live without food."""
        world_area = self.mode.world_size ** 2
        stamina = world_area + len(self.snake) + 1
        stamina = min(world_area * 2, stamina)
        return stamina

    def _is_starved_to_death(self):
        """Return true if the snake hasn't eaten for too long."""
        return self._hunger() > self._stamina()

    def step(self, action):
        """Returns tuple (observation, reward, done, info) as required by gym.Env."""
        self.steps += 1

        reward = 0
        win_reward = 50 * self.reward_unit
        loss_reward = -win_reward

        head_pos = self.snake[0]
        delta = Action.delta(action)
        if action == Action.noop or delta == -self.speed:  # we can't go backwards
            delta = self.speed

        new_head_pos = head_pos + delta
        self.speed = delta

        grow = 0

        crashed = self._is_oob(new_head_pos)
        starved = self._is_starved_to_death()

        if crashed or starved:
            self.over = True
            reward += loss_reward
        else:
            # check whether or not we hit anything
            tile = self.world[new_head_pos.ij]

            if tile == GameTile.snake:
                # hit ourselves
                self.over = True
                reward += loss_reward
            elif tile == GameTile.apple:
                # eat apple
                self.last_meal = self.steps
                self.num_food -= 1
                grow += 1
                reward += 10 * self.reward_unit

        if not self.over:
            # advance snake head
            self.snake.appendleft(new_head_pos)
            self.world[new_head_pos.ij] = GameTile.snake

            # retract snake tail
            retract_tail = 1 - grow
            while len(self.snake) > 1 and retract_tail > 0:
                tail_pos = self.snake.pop()
                self.world[tail_pos.ij] = GameTile.ground
                retract_tail -= 1

            if self._num_vacant_tiles() > 0:
                desired_food = 1
                while self.num_food < desired_food:
                    self._spawn_food()
            else:
                # ate everything
                self.over = True
                reward += win_reward
                logger.info('Finished the game!')

        # clamp reward
        reward = np.clip(reward, *self.reward_range)

        return self._get_state(), reward, self.over, {}

    def _get_state(self):
        """Generate an env observation for the current state."""
        return self.render(mode='rgb_array')

    @staticmethod
    def _tile_border(tile_sz):
        return max(int(tile_sz * 0.1), 1)

    @classmethod
    def _tile_offset(cls, idx, tile_sz):
        return idx * tile_sz + (idx + 1) * cls._tile_border(tile_sz)

    def _world_size(self, tile_sz):
        return self._tile_offset(self.mode.world_size, tile_sz)

    @staticmethod
    def _ui_size(tile_sz):
        return max(tile_sz // 5, 5)

    def _tile_sz_for_resolution(self, res):
        tile_sz = 1
        while self._world_size(tile_sz) < res:
            tile_sz += 1
        return tile_sz

    def _render_ui(self, surface, tile_sz):
        world_size = self._world_size(tile_sz)
        x = world_size
        y = int((self._hunger() / self._stamina()) * world_size)
        rect = pygame.Rect(x, y, self._ui_size(tile_sz), max(0, world_size - y))
        pygame.draw.rect(surface, (0, 255, 0), rect)

    def _render_game_world(self, surface, tile_sz):
        """Render game into a pygame surface."""
        surface.fill((9, 5, 6))
        for (i, j), tile in np.ndenumerate(self.world):
            if tile == GameTile.ground:
                continue

            pos = Vec(i, j)
            if tile == GameTile.snake and self.snake[0] == pos:
                color = GameTile.snake_head_color()
            else:
                color = GameTile.color(tile)
            self.draw_tile(surface, pos, color, tile_sz)

    @classmethod
    def draw_tile(cls, surface, pos, color, tile_sz):
        x = cls._tile_offset(pos.x, tile_sz)
        y = cls._tile_offset(pos.y, tile_sz)
        rect = pygame.Rect(x, y, tile_sz, tile_sz)
        pygame.draw.rect(surface, color, rect)

    def render(self, mode='human'):
        """Implementation of the gym.Env abstract method."""
        if mode not in self.render_modes:
            raise Exception('Unsupported rendering mode')

        if self.rendering_surface[mode] is None:
            screen_size_y = world_size = self._world_size(self.tile_sz[mode])
            screen_size_x = world_size + self._ui_size(self.tile_sz[mode])
            self.rendering_surface[mode] = pygame.Surface((screen_size_x, screen_size_y))
            if mode == 'human' and self.screen is None:
                self.screen = pygame.display.set_mode((screen_size_x, screen_size_y))

        surface = self.rendering_surface[mode]
        self._render_game_world(surface, self.tile_sz[mode])
        self._render_ui(surface, self.tile_sz[mode])

        if mode == 'human':
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        elif mode == 'rgb_array':
            return np.transpose(pygame.surfarray.array3d(surface), axes=[1, 0, 2])

    def close(self):
        """Implementation of the gym.Env abstract method."""
        pygame.quit()

