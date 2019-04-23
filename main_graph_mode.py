"""Graph mode, single GPU

For TensorFlow 1.13
"""

import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from utils import get_celebA, flags
from model import get_generator, get_discriminator

FLAGS = flags.FLAGS
num_tiles = int(np.sqrt(FLAGS.sample_size))

def train():
    z = tf.contrib.distributions.Normal(0., 1.).sample([FLAGS.batch_size, FLAGS.z_dim]) #tf.placeholder(tf.float32, [None, z_dim], name='z_noise')
    ds, images_path = get_celebA(FLAGS.output_size, FLAGS.n_epoch, FLAGS.batch_size)
    iterator = ds.make_one_shot_iterator()
    images = iterator.get_next()

    G = get_generator([None, FLAGS.z_dim])
    D = get_discriminator([None, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])

    G.train()
    D.train()
    fake_images = G(z)
    d_logits = D(fake_images)
    d2_logits = D(images)

    # discriminator: real images are labelled as 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
    # cost for updating discriminator
    d_loss = d_loss_real + d_loss_fake

    # generator: try to make the the fake images look real (1)
    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')
    # Define optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=D.weights)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=G.weights)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    n_step_epoch = int(len(images_path) // FLAGS.batch_size)
    for epoch in range(FLAGS.n_epoch):
        epoch_time = time.time()
        for step in range(n_step_epoch):
            step_time = time.time()
            _d_loss, _g_loss, _, _ = sess.run([d_loss, g_loss, d_optim, g_optim])
            print("Epoch: [{}/{}] [{}/{}] took: {:3f}, d_loss: {:5f}, g_loss: {:5f}".format(epoch, FLAGS.n_epoch, step, n_step_epoch, time.time()-step_time, _d_loss, _g_loss))
            if np.mod(step, FLAGS.save_step) == 0:
                G.save_weights('{}/G.npz'.format(FLAGS.checkpoint_dir), sess=sess, format='npz')
                D.save_weights('{}/D.npz'.format(FLAGS.checkpoint_dir), sess=sess, format='npz')
                result = sess.run(fake_images)
                tl.visualize.save_images(result, [num_tiles, num_tiles], '{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, step))

    sess.close()

if __name__ == '__main__':
    train()
