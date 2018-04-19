from gym.envs.registration import register

from snake_rl.envs.snake import Snake, Action
from snake_rl.envs.wrappers import StackFramesWrapper, ResizeAndGrayscaleWrapper


SNAKE_SIMPLE_LATEST = 'Snake-Simple-v0'
register(
    id=SNAKE_SIMPLE_LATEST,
    entry_point='snake_rl.envs.snake:Snake',
    kwargs={},
)
