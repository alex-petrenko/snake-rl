import sys
import time
import numpy as np

from matplotlib import pyplot as plt

from snake_rl.utils.misc import *
from snake_rl.utils.dnn_utils import *

from snake_rl.algorithms.common import AgentLearner

from snake_rl.algorithms.a2c_vae.replay_memory import ReplayMemory


class VAE:
    def __init__(self, original_obs, num_latent):
        self.num_latent = num_latent

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

        obs_w = obs_h = original_obs.shape[1].value
        obs_ch = original_obs.shape[3].value

        self.encoder_conv_size = self.encoder_flat_size = -1
        self.encoder_channels = 256

        prior = self.make_prior()
        posterior = self.encoder(original_obs)

        mu, sigma = posterior.mean(), posterior.stddev()

        self.z = posterior.sample()

        self.generated_obs, output_distribution = self.decoder(self.z, obs_w, obs_h, obs_ch)

        likelihood = output_distribution.log_prob(original_obs)
        divergence = tf.distributions.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)

        generated_obs_flat = tf.reshape(self.generated_obs, [-1, obs_w * obs_h * obs_ch])
        obs_flat = tf.reshape(original_obs, [-1, obs_w * obs_h * obs_ch])
        img_loss = tf.reduce_sum(tf.squared_difference(generated_obs_flat, obs_flat), 1)
        img_loss = tf.reduce_mean(img_loss)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = img_loss + regularization_loss

        use_elbo = False  # if False then we're basically training a regular autoencoder
        if use_elbo:
            self.loss += -elbo

        self._summary_key = 'vae'
        with tf.name_scope('vae_summary'):
            self._scalar_summary('vae_likelihood', tf.reduce_mean(likelihood))
            self._scalar_summary('vae_divergence', tf.reduce_mean(divergence))
            self._scalar_summary('vae_img_loss', img_loss)
            self._scalar_summary('vae_reg_loss', regularization_loss)
            self._scalar_summary('vae_loss', self.loss)
            self._scalar_summary('vae_avg_mu', tf.reduce_mean(mu))
            self._scalar_summary('vae_avg_sigma', tf.reduce_mean(sigma))
            self.summaries = tf.summary.merge_all(key=self._summary_key)

    def _scalar_summary(self, name, tensor):
        return tf.summary.scalar(name, tensor, collections=[self._summary_key])

    def make_prior(self):
        mu = tf.zeros(self.num_latent)
        sigma = tf.ones(self.num_latent)
        return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)

    def encoder(self, original_obs):
        with tf.variable_scope('encoder', reuse=None):
            x = conv(original_obs, 32, 3, 2, self.regularizer)
            x = conv(x, 64, 3, 2, self.regularizer)
            x = conv(x, 64, 3, 2, self.regularizer)
            x = conv(x, 128, 3, 2, self.regularizer)
            x = conv(x, self.encoder_channels, 3, 2, self.regularizer)
            self.encoder_conv_size = x.shape[1].value

            x = tf.contrib.layers.flatten(x)
            self.encoder_flat_size = x.shape[1].value

            mu = dense(x, self.num_latent)
            sigma = dense(x, self.num_latent, activation=tf.nn.softplus)  # softplus is log(exp(x) + 1)
            return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)

    def decoder(self, sampled_z, obs_w, obs_h, obs_ch):
        with tf.variable_scope('decoder', reuse=None):
            x = dense(sampled_z, self.num_latent * 2, self.regularizer)
            x = dense(x, self.encoder_flat_size, self.regularizer)

            x = tf.reshape(x, [-1, self.encoder_conv_size, self.encoder_conv_size, self.encoder_channels])
            x = conv_t(x, self.encoder_channels, 3, 2)
            x = conv_t(x, 128, 3, 2)
            x = conv_t(x, 64, 3, 2)
            x = conv_t(x, 32, 3, 2)
            x = conv_t(x, 3, 3, 2, activation=tf.nn.sigmoid)

            img = tf.reshape(x, shape=[-1, obs_w, obs_h, obs_ch])

            img_distribution = tf.contrib.distributions.Bernoulli(img)
            # img = img_distribution.probs
            img_distribution = tf.contrib.distributions.Independent(img_distribution, 3)
            return img, img_distribution


class Autoencoder:
    """Regular (non-variational) autoencoder for comparison."""

    def __init__(self, original_obs, num_latent):
        self.num_latent = num_latent

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

        obs_w = obs_h = original_obs.shape[1].value
        obs_ch = original_obs.shape[3].value

        self.encoder_conv_size = self.encoder_flat_size = -1
        self.encoder_channels = 128
        self.z = self.encoder(original_obs)

        self.generated_obs = self.decoder(self.z, obs_w, obs_h, obs_ch)

        generated_obs_flat = tf.reshape(self.generated_obs, [-1, obs_w * obs_h * obs_ch])
        obs_flat = tf.reshape(original_obs, [-1, obs_w * obs_h * obs_ch])
        img_loss = tf.reduce_sum(tf.squared_difference(generated_obs_flat, obs_flat), 1)
        img_loss = tf.reduce_mean(img_loss)

        regularization_loss = tf.losses.get_regularization_loss()
        self.loss = img_loss + regularization_loss

        self._summary_key = 'autoenc'
        with tf.name_scope('autoenc_summary'):
            self._scalar_summary('img_loss', img_loss)
            self._scalar_summary('reg_loss', regularization_loss)
            self._scalar_summary('z', tf.reduce_mean(tf.abs(self.z)))
            self._scalar_summary('autoenc_loss', self.loss)
            self.summaries = tf.summary.merge_all(key=self._summary_key)

    def _scalar_summary(self, name, tensor):
        return tf.summary.scalar(name, tensor, collections=[self._summary_key])

    def encoder(self, original_obs):
        with tf.variable_scope('encoder', reuse=None):
            x = conv(original_obs, 32, 3, 2, self.regularizer)
            x = conv(x, 64, 3, 2, self.regularizer)
            x = conv(x, self.encoder_channels, 3, 2, self.regularizer)
            self.encoder_conv_size = x.shape[1].value

            x = tf.contrib.layers.flatten(x)
            self.encoder_flat_size = x.shape[1].value

            x = dense(x, self.num_latent * 2, self.regularizer)
            z = dense(x, self.num_latent, self.regularizer, activation=None)
            return z

    def decoder(self, z, obs_w, obs_h, obs_ch):
        with tf.variable_scope('decoder', reuse=None):
            x = dense(z, self.num_latent * 2, self.regularizer)
            x = dense(x, self.encoder_flat_size, self.regularizer)

            x = tf.reshape(x, [-1, self.encoder_conv_size, self.encoder_conv_size, self.encoder_channels])
            x = conv_t(x, 64, 3, 2)
            x = conv_t(x, 32, 3, 2)
            x = conv_t(x, 3, 3, 2, activation=None)

            img = tf.reshape(x, shape=[-1, obs_w, obs_h, obs_ch])
            return img


class Policy:
    class CategoricalProbabilityDistribution:
        """Based on https://github.com/openai/baselines implementation."""

        def __init__(self, logits):
            self.logits = logits

        def entropy(self):
            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        def sample(self):
            u = tf.random_uniform(tf.shape(self.logits))
            return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def __init__(self, embedding, num_actions, regularizer):
        # fully-connected layers to generate actions
        fc = dense(embedding, 512, regularizer)
        fc = dense(fc, 512, regularizer)
        actions_fc = dense(fc, 256, regularizer)
        self.actions = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.best_action_deterministic = tf.argmax(self.actions, axis=1)
        self.actions_prob_distribution = Policy.CategoricalProbabilityDistribution(self.actions)
        self.act = self.actions_prob_distribution.sample()

        value_fc = dense(fc, 256, regularizer)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])


class AgentA2CVae(AgentLearner):
    class Params(AgentLearner.Params):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentA2CVae.Params, self).__init__(experiment_name)
            # A2C algorithm parameters
            self.rollout = 5  # number of successive env steps used for each model update
            self.num_envs = 16  # number of environments running in parallel. Batch size = rollout * num_envs
            self.gamma = 0.98  # future reward discount

            # encoder
            self.num_latent = 128
            self.use_vae = False
            self.enc_train_batch = 128
            self.enc_train_ratio = 10

            # components of the A2C loss function
            self.initial_entropy_loss_coeff = 1.0
            self.value_loss_coeff = 0.5

            # training
            self.enc_learning_rate = self.a2c_learning_rate = 1e-4
            self.train_for_steps = 100000
            self.save_every = 500
            self.summaries_every = 100
            self.print_every = 50
            self.viz_every = 500

    def __init__(self, env, params):
        """Initialize A2C computation graph and some auxiliary tensors."""
        super(AgentA2CVae, self).__init__(params)

        self.best_avg_reward = -sys.float_info.max

        num_actions = env.action_space.n

        global_step = tf.train.get_or_create_global_step()

        input_shape = list(env.observation_space.shape)
        input_shape = [None] + input_shape  # add batch dimension
        self.observations = tf.placeholder(tf.float32, shape=input_shape)

        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

        self.memory = ReplayMemory()

        if self.params.use_vae:
            self.autoencoder = VAE(self.observations, params.num_latent)
        else:
            self.autoencoder = Autoencoder(self.observations, params.num_latent)

        self.sampled_embeddings = tf.placeholder(tf.float32, [None, self.autoencoder.num_latent])

        self.policy = Policy(self.sampled_embeddings, num_actions, regularizer)

        self.selected_actions = tf.placeholder(tf.int32, [None])  # action selected by the policy
        self.value_estimates = tf.placeholder(tf.float32, [None])
        self.discounted_rewards = tf.placeholder(tf.float32, [None])  # estimate of total reward (rollout + value)

        advantages = self.discounted_rewards - self.value_estimates

        # negative logarithms of the probabilities of actions
        # softmax turns action logits into a probability distribution over actions
        # cross entropy with true distribution (one-hot vector with 1 for action taken and 0 everywhere else)
        # will give h(p,q) = -sum_over_actions(p(action) * log q(action)) = -p(taken) * log q(taken) = -log q(taken)
        neglogp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.policy.actions, labels=self.selected_actions,
        )

        # maximize probabilities of actions that give high advantage
        action_loss = tf.reduce_mean(advantages * neglogp_actions)

        # penalize for inaccurate value estimation
        value_loss = tf.losses.mean_squared_error(self.discounted_rewards, self.policy.value)
        value_loss = self.params.value_loss_coeff * value_loss

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_loss = -tf.reduce_mean(self.policy.actions_prob_distribution.entropy())
        entropy_loss_coeff = tf.train.exponential_decay(
            self.params.initial_entropy_loss_coeff, tf.cast(global_step, tf.float32), 50.0, 0.95, staircase=True,
        )
        entropy_loss_coeff = tf.maximum(entropy_loss_coeff, 0.02)
        entropy_loss = entropy_loss_coeff * entropy_loss

        a2c_loss = action_loss + entropy_loss + value_loss
        regularization_loss = tf.losses.get_regularization_loss()
        loss = regularization_loss + a2c_loss

        # training
        autoenc_optimizer = tf.train.AdamOptimizer(learning_rate=self.params.enc_learning_rate)
        self.train_enc = autoenc_optimizer.minimize(self.autoencoder.loss)
        a2c_optimizer = tf.train.AdamOptimizer(learning_rate=self.params.a2c_learning_rate)
        self.train_a2c = a2c_optimizer.minimize(loss, global_step=global_step)

        # summaries for the agent and the training process
        with tf.name_scope('agent_summary'):
            tf.summary.histogram('actions', self.policy.actions)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.policy.act)))
            tf.summary.scalar('advantages', tf.reduce_mean(advantages))
            tf.summary.scalar('value', tf.reduce_mean(self.policy.value))

            tf.summary.histogram('selected_actions', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.actions_prob_distribution.entropy()))
            tf.summary.scalar('entropy_coeff', entropy_loss_coeff)

            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('entropy_loss', entropy_loss)
            tf.summary.scalar('a2c_loss', a2c_loss)
            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('loss', loss)

            self.a2c_summaries = tf.summary.merge_all()

        logger.info('Total parameters in the model: %d', count_total_parameters())

        summary_dir = ensure_dir_exists(join(summaries_dir(), self.params.experiment_name))
        self.summary_writer = tf.summary.FileWriter(summary_dir)
        self.saver = tf.train.Saver(max_to_keep=3)

    def _maybe_print(self, step, avg_rewards, fps):
        if step % self.params.print_every == 0:
            logger.info('<====== Step %d ======>', step)
            logger.info('FPS: %.1f', fps)
            if avg_rewards > self.best_avg_reward:
                self.best_avg_reward = avg_rewards
                logger.info('<<<<< New record! %.3f >>>>>\n', self.best_avg_reward)
            logger.info('Avg. 100 episode reward: %.3f (best: %.3f)', avg_rewards, self.best_avg_reward)

    def _maybe_visualize_encoder(self, step, observations):
        if step % self.params.viz_every == 0:
            generated_obs = self.session.run(
                self.autoencoder.generated_obs, feed_dict={self.observations: observations},
            )

            fig = plt.figure(figsize=(10, 4))
            row = 2
            col = len(observations)

            i = 0
            for imgs in [observations, generated_obs]:
                for img in imgs:
                    i += 1
                    fig.add_subplot(row, col, i)
                    plt.axis('off')
                    plt.imshow(img)

            viz_dir = ensure_dir_exists(join(experiment_dir(self.params.experiment_name), '.viz'))
            plt.savefig(join(viz_dir, '{:07}_enc.png'.format(step)))
            plt.close()

    def best_action(self, observation, deterministic=False):
        show_generated_observations = False
        embeddings, generated_obs = self.session.run(
            [self.autoencoder.z, self.autoencoder.generated_obs], feed_dict={self.observations: [observation]},
        )

        if show_generated_observations:
            plt.imshow(observation)
            plt.show()
            plt.imshow(generated_obs[0])
            plt.show()

        embeddings = self._calc_embeddings([observation])
        actions, _ = self._policy_step(embeddings, deterministic)
        logger.info('Best selected action %d, for an embedding: %r...', actions[0], list(embeddings[0][:4]))
        return actions[0]

    def _calc_embeddings(self, observations):
        return self.session.run(self.autoencoder.z, feed_dict={self.observations: observations})

    def _train_enc_step(self, step, observations, enable_summaries):
        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        with_summaries = with_summaries and enable_summaries
        summaries = [self.autoencoder.summaries] if with_summaries else []
        result = self.session.run(
            [self.train_enc] + summaries,
            feed_dict={self.observations: observations},
        )

        if with_summaries:
            summary = result[-1]
            self.summary_writer.add_summary(summary, global_step=step)

    def _policy_step(self, embeddings, deterministic=False):
        """
        Select the best action by sampling from the distribution generated by the policy. Also estimate the
        value for the currently observed environment state.
        """
        ops = [
            self.policy.best_action_deterministic if deterministic else self.policy.act,
            self.policy.value,
        ]
        actions, values = self.session.run(ops, feed_dict={self.sampled_embeddings: embeddings})
        return actions, values

    def _estimate_values(self, embeddings):
        values = self.session.run(
            self.policy.value,
            feed_dict={self.sampled_embeddings: embeddings},
        )
        return values

    def _train_a2c_step(self, step, embeddings, actions, values, discounted_rewards):
        """
        Actually do a single iteration of training. See the computational graph in the ctor to figure out
        the details.
        """
        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        summaries = [self.a2c_summaries] if with_summaries else []
        result = self.session.run(
            [self.train_a2c] + summaries,
            feed_dict={
                self.sampled_embeddings: embeddings,
                self.selected_actions: actions,
                self.value_estimates: values,
                self.discounted_rewards: discounted_rewards,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[-1]
            self.summary_writer.add_summary(summary, global_step=step)

        return step

    @staticmethod
    def _calc_discounted_rewards(gamma, rewards, dones, last_value):
        """Calculate gamma-discounted rewards for an n-step A2C."""
        cumulative = 0 if dones[-1] else last_value
        discounted_rewards = []
        for rollout_step in reversed(range(len(rewards))):
            r, done = rewards[rollout_step], dones[rollout_step]
            cumulative = r + gamma * cumulative * (not done)
            discounted_rewards.append(cumulative)
        return reversed(discounted_rewards)

    def learn(self, multi_env, step_callback=None):
        step = initial_step = tf.train.global_step(self.session, tf.train.get_global_step())
        training_started = time.time()
        batch_size = self.params.rollout * self.params.num_envs

        observations = multi_env.initial_observations()
        embeddings = self._calc_embeddings(observations)

        end_of_training = lambda s: s >= self.params.train_for_steps
        while not end_of_training(step):
            batch_obs = [observations]
            batch_embeddings = [embeddings]
            batch_actions, batch_values, batch_rewards, batch_dones = [], [], [], []
            for rollout_step in range(self.params.rollout):

                actions, values = self._policy_step(embeddings)
                batch_actions.append(actions)
                batch_values.append(values)

                # wait for all the workers to complete an environment step
                observations, rewards, dones = multi_env.step(actions)
                embeddings = self._calc_embeddings(observations)
                batch_rewards.append(rewards)
                batch_dones.append(dones)

                if rollout_step != self.params.rollout - 1:
                    # we don't need the newest observation in the training batch, already have enough
                    batch_obs.append(observations)
                    batch_embeddings.append(embeddings)

            assert len(batch_obs) == len(batch_rewards)

            batch_rewards = np.asarray(batch_rewards, np.float32).swapaxes(0, 1)
            batch_dones = np.asarray(batch_dones, np.bool).swapaxes(0, 1)
            last_values = self._estimate_values(embeddings)

            gamma = self.params.gamma
            discounted_rewards = []
            for env_rewards, env_dones, last_value in zip(batch_rewards, batch_dones, last_values):
                discounted_rewards.extend(self._calc_discounted_rewards(gamma, env_rewards, env_dones, last_value))

            # convert observations and estimations to meaningful n-step batches
            batch_obs_shape = (self.params.rollout * multi_env.num_envs,) + observations[0].shape
            batch_obs = np.asarray(batch_obs, np.float32).swapaxes(0, 1).reshape(batch_obs_shape)

            batch_embeddings_shape = (self.params.rollout * multi_env.num_envs,) + embeddings[0].shape
            batch_embeddings = np.asarray(batch_embeddings, np.float32).swapaxes(0, 1).reshape(batch_embeddings_shape)

            batch_actions = np.asarray(batch_actions, np.int32).swapaxes(0, 1).flatten()
            batch_values = np.asarray(batch_values, np.float32).swapaxes(0, 1).flatten()

            self.memory.remember(batch_obs)

            for i in range(self.params.enc_train_ratio):
                if self.memory.len_memory() > self.params.enc_train_batch:
                    self._train_enc_step(step, self.memory.recollect(self.params.enc_train_batch), i == 0)

            step = self._train_a2c_step(step, batch_embeddings, batch_actions, batch_values, discounted_rewards)

            self._maybe_save(step)

            avg_rewards = multi_env.calc_avg_rewards(n=100)
            fps = ((step - initial_step) * batch_size) / (time.time() - training_started)
            self._maybe_print(step, avg_rewards, fps)
            self._maybe_visualize_encoder(step, batch_obs[:8])
            if step_callback is not None:
                step_callback(locals(), globals())
