import gym
import sys
import logging

from snake_rl.utils.misc import *
from snake_rl.utils import init_logger, Monitor

from snake_rl.envs.wrappers import wrap_env

from snake_rl.algorithms.baselines import a2c
from snake_rl.algorithms.baselines.a2c.a2c_utils import *


logger = logging.getLogger(os.path.basename(__file__))


class A2CMonitor(Monitor):
    def callback(self, local_vars, _):
        time_step = local_vars['step']
        if time_step % 10 == 0:
            self.progress_file.write('{},{}\n'.format(time_step, local_vars['avg_rewards']))
            self.progress_file.flush()


def train(a2c_params, env_id):
    multithread_env = a2c.MultiEnv(a2c_params.num_envs, make_env_func=lambda: wrap_env(gym.make(env_id)))

    agent = a2c.AgentA2C(multithread_env, params=a2c_params)
    agent.initialize()

    with A2CMonitor(a2c_params.experiment_name) as monitor:
        agent.learn(multithread_env, step_callback=monitor.callback)

    agent.finalize()
    multithread_env.close()
    return 0


def main():
    init_logger()

    env_id = CURRENT_ENV
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)

    params = a2c.AgentA2C.Params(experiment)
    params.learning_rate = 1e-4
    params.gamma = 0.98
    params.entropy_loss_coeff = 0.11
    params.rollout = 10
    params.num_envs = 16
    params.train_for_steps = 200000
    return train(params, env_id)


if __name__ == '__main__':
    sys.exit(main())
