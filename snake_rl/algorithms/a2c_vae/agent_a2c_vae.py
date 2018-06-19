import numpy as np

from snake_rl.utils.misc import *
from snake_rl.utils.dnn_utils import *

from snake_rl.algorithms.common import AgentLearner


class AgentA2CVae(AgentLearner):
    class Params(AgentLearner.Params):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentA2CVae.Params, self).__init__(experiment_name)

            self.num_envs = 16  # number of environments running in parallel. Batch size = rollout * num_envs

    def __init__(self, env, params):
        """Initialize A2C computation graph and some auxiliary tensors."""
        super(AgentA2CVae, self).__init__(params)

    def best_action(self, observation):
        pass

    def _train_step(self, step, observations, actions, values, discounted_rewards):
        """
        Actually do a single iteration of training. See the computational graph in the ctor to figure out
        the details.
        """
        pass

    @staticmethod
    def _calc_discounted_rewards(gamma, rewards, dones, last_value):
        """Calculate gamma-discounted rewards for an n-step A2C."""
        pass

    def learn(self, multi_env, step_callback=None):
        pass
