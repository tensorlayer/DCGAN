"""Eager mode, single GPU
"""

import os, time, multiprocessing
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorlayer as tl
from glob import glob
from utils import get_celebA, flags # get_image
from model import get_generator, get_discriminator

FLAGS = flags.FLAGS
num_tiles = int(np.sqrt(FLAGS.sample_size))

def correct_grad(grad, scale):
    if grad != None:
        return grad * scale
    else:
        return None
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def train():
    images, images_path = get_celebA(FLAGS.output_size, FLAGS.n_epoch, FLAGS.batch_size)
    G = get_generator([None, FLAGS.z_dim])
    D = get_discriminator([None, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])

    G.train()
    D.train()

    d_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    g_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)

    n_step_epoch = int(len(images_path) // FLAGS.batch_size)

    for step, batch_images in enumerate(images):
        step_time = time.time()

        with tf.GradientTape(persistent=True) as tape:
            z = tf.contrib.distributions.Normal(0., 1.).sample([FLAGS.batch_size, FLAGS.z_dim]) #tf.placeholder(tf.float32, [None, z_dim], name='z_noise')
            d_logits = D(G(z))
            d2_logits = D(batch_images)
            d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='real')
            d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='fake')

        grad_gd = tape.gradient(d_loss_fake, G.weights+D.weights)
        grad_d1 = tape.gradient(d_loss_real, D.weights)
        scale = -1 #tf.reduce_mean(sigmoid(d_logits)/(sigmoid(d_logits)-1))
        grad_g = grad_gd[0:len(G.weights)]
        for i in range(len(grad_g)):
            if grad_g[i]!=None: # batch_norm moving mean, var
                grad_g[i] = grad_g[i] * scale
            # grad_d1 = list(filter(lambda x: correct_grad(x, scale), grad_d1))
        grad_d2 = grad_gd[len(G.weights):]
        grad_d = []
        for x,y in zip(grad_d1, grad_d2):
            if x==None: # batch_norm moving mean, var
                grad_d.append(None)
            else:
                grad_d.append(x+y)
        g_optimizer.apply_gradients(zip(grad_g, G.weights))
        d_optimizer.apply_gradients(zip(grad_d, D.weights))
        del tape

        g_loss = d_loss_fake
        d_loss = d_loss_real+d_loss_fake

        print("Epoch: [{}/{}] [{}/{}] took: {:3f}, d_loss: {:5f}, g_loss: {:5f}".format(step//n_step_epoch, FLAGS.n_epoch, step, n_step_epoch, time.time()-step_time, d_loss, g_loss))
        if np.mod(step, FLAGS.save_step) == 0:
            G.save_weights('{}/G.npz'.format(FLAGS.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(FLAGS.checkpoint_dir), format='npz')
            result = G(z)
            tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles], '{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, step//n_step_epoch, step))

if __name__ == '__main__':
    train()
    # try:
    #     tf.app.run()
    # except KeyboardInterrupt:
    #     print('EXIT')
