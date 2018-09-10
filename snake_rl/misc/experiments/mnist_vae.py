"""
Experimented with simple VAE, decided to keep the code.
Heavily inspired by https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb

"""


import sys
import numpy as np

from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from snake_rl.utils import init_logger
from snake_rl.utils.misc import experiment_dir, model_dir

from snake_rl.utils.dnn_utils import *

logger = logging.getLogger(os.path.basename(__file__))


EXPERIMENT = 'mnist_vae_experiment'
N_LATENT = 20


def make_prior():
    mu = tf.zeros(N_LATENT)
    sigma = tf.ones(N_LATENT)
    return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)


def make_encoder(x_input):
    x_input = tf.reshape(x_input, shape=[-1, 28, 28, 1])
    x = conv(x_input, 32, 3, 2)
    x = conv(x, 64, 3, 2)
    x = conv(x, 128, 3, 2)
    x = tf.contrib.layers.flatten(x)
    mu = dense(x, N_LATENT)
    sigma = dense(x, N_LATENT, activation=tf.nn.softplus)  # softplus is log(exp(x) + 1)
    return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)


def make_mlp_encoder(x_input):
    x = tf.layers.flatten(x_input)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    loc = tf.layers.dense(x, N_LATENT)
    scale = tf.layers.dense(x, N_LATENT, tf.nn.softplus)
    return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)


def make_decoder(sampled_z):
    x = tf.layers.dense(sampled_z, 24, tf.nn.relu)
    x = tf.layers.dense(x, 7 * 7 * 64, tf.nn.relu)
    x = tf.reshape(x, [-1, 7, 7, 64])

    x = tf.layers.conv2d_transpose(x, 64, 3, 2, 'SAME', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 32, 3, 2, 'SAME', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 1, 3, 1, 'SAME')

    img = tf.reshape(x, [-1, 28, 28])

    img_distribution = tf.contrib.distributions.Bernoulli(img)
    img = img_distribution.probs
    img_distribution = tf.contrib.distributions.Independent(img_distribution, 2)
    return img, img_distribution


def make_mlp_decoder(sampled_z, data_shape):
    x = sampled_z
    x = tf.layers.dense(x, 200, tf.nn.relu)
    x = tf.layers.dense(x, 200, tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)

    img_distribution = tf.contrib.distributions.Bernoulli(logit)
    img = img_distribution.probs
    img_distribution = tf.contrib.distributions.Independent(img_distribution, 2)
    return img, img_distribution


def main():
    """Script entry point."""
    init_logger()

    mnist = input_data.read_data_sets(os.path.join(experiment_dir(EXPERIMENT), 'MNIST_data'))

    tf.reset_default_graph()

    batch_size = 128

    x_input = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')

    prior = make_prior()
    posterior = make_encoder(x_input)

    mu, sigma = posterior.mean(), posterior.stddev()

    z = posterior.sample()

    generated_img, output_distribution = make_decoder(z)

    likelihood = output_distribution.log_prob(x_input)
    divergence = tf.distributions.kl_divergence(posterior, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    loss = -elbo

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

    with tf.Session() as session:
        saver = tf.train.Saver(max_to_keep=3)
        checkpoint_dir = model_dir(EXPERIMENT)
        try:
            saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            logger.info('Didn\'t find a valid restore point, start from scratch')
            session.run(tf.global_variables_initializer())

        batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
        i = tf.train.global_step(session, tf.train.get_global_step())
        for epoch in range(20):
            values = session.run(
                [generated_img, loss, likelihood, divergence, mu, sigma],
                feed_dict={x_input: batch},
            )

            # unpack the tuple
            img, ls, likelihood_ls, kl_ls, mu_val, sigma_val = values

            plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
            plt.show()
            plt.imshow(img[0], cmap='gray')
            plt.show()
            logger.info(
                'i: %d, loss: %f, likeli: %f, kl_ls: %f, mu: %f, sigma: %f',
                i, ls, np.mean(likelihood_ls), np.mean(kl_ls), np.mean(mu_val), np.mean(sigma_val),
            )

            for _ in range(200):
                batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
                session.run(optimizer, feed_dict={x_input: batch})

            i = tf.train.global_step(session, tf.train.get_global_step())

            logger.info('Step #%d, saving...', i)
            saver_path = checkpoint_dir + '/model'
            saver.save(session, saver_path, global_step=i)

            logger.info('End of epoch %d, step %d', epoch, i)

        fig = plt.figure(figsize=(8, 8))
        row = col = 5

        randoms = [np.random.normal(0, 1, N_LATENT) for _ in range(col * row)]
        imgs = session.run(generated_img, feed_dict={z: randoms})
        imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
        for i, img in enumerate(imgs):
            fig.add_subplot(row, col, i + 1)
            plt.axis('off')
            plt.imshow(img, cmap='gray')

        logger.info('Press any key to exit')
        plt.show()

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
