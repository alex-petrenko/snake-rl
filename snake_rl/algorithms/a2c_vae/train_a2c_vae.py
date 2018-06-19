import gym
import sys
import logging

from snake_rl.utils.misc import *
from snake_rl.utils import init_logger, Monitor

from snake_rl.envs.wrappers import wrap_env

from snake_rl.algorithms import a2c_vae
from snake_rl.algorithms.a2c_vae.a2c_vae_utils import *
from snake_rl.algorithms.utils.multi_env import MultiEnv


logger = logging.getLogger(os.path.basename(__file__))


class A2CVaeMonitor(Monitor):
    def callback(self, local_vars, _):
        time_step = local_vars['step']
        if time_step % 10 == 0:
            self.progress_file.write('{},{}\n'.format(time_step, local_vars['avg_rewards']))
            self.progress_file.flush()


def train(params, env_id):
    multithread_env = MultiEnv(params.num_envs, make_env_func=lambda: wrap_env(gym.make(env_id)))

    agent = a2c_vae.AgentA2CVae(multithread_env, params=params)
    agent.initialize()

    with A2CVaeMonitor(params.experiment_name) as monitor:
        agent.learn(multithread_env, step_callback=monitor.callback)

    agent.finalize()
    multithread_env.close()
    return 0


def main():
    init_logger()

    env_id = CURRENT_ENV
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)

    params = a2c_vae.AgentA2CVae.Params(experiment)
    return train(params, env_id)


if __name__ == '__main__':
    sys.exit(main())
