"""
Play a version of the environment with human controls.

"""


import os
import gym
import sys
import time
import pygame
import logging

from snake_rl import envs
from snake_rl.envs.snake import Action

from snake_rl.utils import init_logger


logger = logging.getLogger(os.path.basename(__file__))


def main():
    """Script entry point."""
    init_logger()

    env = gym.make(envs.SNAKE_SIMPLE_LATEST)

    input_fps = 100
    input_interval = 1.0 / input_fps
    snake_fps = 5
    snake_interval = 1.0 / snake_fps
    last_snake_tick = time.time()

    episode_rewards = []
    while not env.should_quit():
        action = prev_action = Action.noop
        env.reset()
        env.render()

        done = False
        episode_reward = 0.0
        while not done:
            while True:
                last_action = env.process_events()
                until_next_tick = (last_snake_tick + snake_interval) - time.time()
                if last_action != Action.noop:
                    action = last_action
                    if action != prev_action:
                        wait_time_ms = max(until_next_tick / 2, input_interval) * 1000
                        pygame.time.wait(int(wait_time_ms))
                        break

                if until_next_tick < 1e-5 or env.should_quit():
                    break
                wait_time_ms = min(until_next_tick, input_interval) * 1000
                pygame.time.wait(int(wait_time_ms))

            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            prev_action = action
            last_snake_tick = time.time()

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        logger.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

        if not env.should_quit():
            # display the end position in the game for a couple of sec
            for _ in range(snake_fps):
                env.render()
                env.clock.tick(snake_fps)
                env.process_events()

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
