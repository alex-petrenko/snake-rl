from gym.envs.registration import register

from snake_rl.envs.snake import Snake, Action, GameMode
from snake_rl.envs.wrappers import StackFramesWrapper, ResizeAndGrayscaleWrapper


SNAKE_TINY_LATEST = 'Snake-Tiny-v0'
register(
    id=SNAKE_TINY_LATEST,
    entry_point='snake_rl.envs.snake:Snake',
    kwargs={'mode': GameMode.snake_tiny()},
)

SNAKE_SIMPLE_LATEST = 'Snake-Simple-v0'
register(
    id=SNAKE_SIMPLE_LATEST,
    entry_point='snake_rl.envs.snake:Snake',
    kwargs={},
)
