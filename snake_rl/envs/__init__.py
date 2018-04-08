from gym.envs.registration import register


SNAKE_SIMPLE_LATEST = 'Snake-Simple-v0'
register(
    id=SNAKE_SIMPLE_LATEST,
    entry_point='snake_rl.envs.snake:Snake',
    kwargs={},
)
