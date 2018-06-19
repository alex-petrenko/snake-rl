"""
Experimented with simple VAE, decided to keep the code.
Heavily inspired by https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb

"""


import os
import sys
import logging
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from snake_rl.utils import init_logger
from snake_rl.utils.misc import experiment_dir, model_dir

logger = logging.getLogger(os.path.basename(__file__))


EXPERIMENT = 'mnist_vae_experiment'
N_LATENT = 8


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def conv(x, filters, kernel_size, stride=1, regularizer=None):
    return tf.contrib.layers.conv2d(
        x,
        filters,
        kernel_size,
        stride=stride,
        padding='same',
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
        activation_fn=lrelu,
    )


def conv_t(x, filters, kernel_size, stride=1, regularizer=None):
    return tf.layers.conv2d_transpose(
        x,
        filters,
        kernel_size,
        stride,
        padding='same',
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu,
    )


def encoder(X_in, keep_prob):
    with tf.variable_scope('encoder', reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = conv(X, 64, 4, 2)
        x = tf.nn.dropout(x, keep_prob)
        x = conv(x, 64, 4, 2)
        x = tf.nn.dropout(x, keep_prob)
        x = conv(x, 64, 4, 1)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mu = tf.layers.dense(x, units=N_LATENT)
        log_sigma = tf.layers.dense(x, units=N_LATENT)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], N_LATENT]))
        z = mu + epsilon * tf.exp(log_sigma / 2)
        return z, mu, log_sigma


def decoder(sampled_z, keep_prob):
    with tf.variable_scope('decoder', reuse=None):
        x = tf.layers.dense(sampled_z, units=24, activation=lrelu)

        # final encoder resolution = 7x7, let's have about as many dimensions at decoder input
        x = tf.layers.dense(x, units=49, activation=lrelu)

        x = tf.reshape(x, [-1, 7, 7, 1])
        x = conv_t(x, 64, 4, 2)
        x = tf.nn.dropout(x, keep_prob)
        x = conv_t(x, 64, 4, 1)
        x = tf.nn.dropout(x, keep_prob)
        x = conv_t(x, 64, 4, 1)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


def main():
    """Script entry point."""
    init_logger()

    mnist = input_data.read_data_sets(os.path.join(experiment_dir(EXPERIMENT), 'MNIST_data'))

    tf.reset_default_graph()

    batch_size = 64

    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

    z, mu, log_sigma = encoder(X, keep_prob)
    generated_img = decoder(z, keep_prob)

    # loss
    generated_img_flat = tf.reshape(generated_img, [-1, 28 * 28])
    input_flat = tf.reshape(X, [-1, 28 * 28])
    img_loss = tf.reduce_sum(tf.squared_difference(generated_img_flat, input_flat))
    kl_divergence_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * log_sigma - tf.square(mu) - tf.exp(2.0 * log_sigma), 1)
    loss = tf.reduce_mean(img_loss + kl_divergence_loss)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss, global_step=global_step)

    with tf.Session() as session:
        saver = tf.train.Saver(max_to_keep=3)
        checkpoint_dir = model_dir(EXPERIMENT)
        try:
            saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            logger.info('Didn\'t find a valid restore point, start from scratch')
            session.run(tf.global_variables_initializer())

        while True:
            i = tf.train.global_step(session, tf.train.get_global_step())
            if i > 35000:
                break

            batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
            session.run(optimizer, feed_dict={X: batch, keep_prob: 0.8})

            if i % 200 == 0:
                logger.info('iteration: #%d', i)

            if i % 5000 == 0:
                values = session.run(
                    [loss, generated_img, img_loss, kl_divergence_loss, mu, log_sigma],
                    feed_dict={X: batch, keep_prob: 1.0},
                )

                # unpack the tuple
                ls, img, im_ls, kl_ls, mu_val, log_sigma_val = values

                plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
                plt.show()
                plt.imshow(img[0], cmap='gray')
                plt.show()
                logger.info(
                    'i: %d ls: %f im_ls: %f kl_ls: %f mu: %f log_sigma: %f',
                    i, ls, np.mean(im_ls), np.mean(kl_ls), np.mean(mu_val), np.mean(log_sigma_val),
                )

                logger.info('Step #%d, saving...', i)
                saver_path = checkpoint_dir + '/model'
                saver.save(session, saver_path, global_step=i)

        fig = plt.figure(figsize=(8, 8))
        row = col = 5

        randoms = [np.random.normal(0, 1, N_LATENT) for _ in range(col * row)]
        imgs = session.run(generated_img, feed_dict={z: randoms, keep_prob: 1.0})
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
