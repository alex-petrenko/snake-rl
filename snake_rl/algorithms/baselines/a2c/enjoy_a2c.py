"""
Execute the learned policy.

"""


import gym
import sys
import logging

from snake_rl.utils.misc import *
from snake_rl.utils import init_logger

from snake_rl.envs.wrappers import wrap_env

from snake_rl.algorithms.common import run_policy_loop

from snake_rl.algorithms.baselines import a2c
from snake_rl.algorithms.baselines.a2c.a2c_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def enjoy(experiment, env_id, max_num_episodes=1000000, fps=7):
    env = wrap_env(gym.make(env_id))
    env.seed(0)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    return run_policy_loop(agent, env, max_num_episodes, fps)


def main():
    init_logger()
    env_id = CURRENT_ENV
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)
    return enjoy(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
