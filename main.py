""" TensorLayer implementation of Deep Convolutional Generative Adversarial Network (DCGAN).
Using deep convolutional generative adversarial networks (DCGAN) 
to generate face images from a noise distribution.
References:
    -Generative Adversarial Nets.
    Goodfellow et al. arXiv: 1406.2661.
    - Unsupervised Representation Learning with Deep Convolutional 
    Generative Adversarial Networks. A Radford, L Metz, S Chintala. 
    arXiv: 1511.06434.
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf)
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434)
Usage:
    - See README.md
"""
import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from glob import glob
from random import shuffle

from model import generator_simplified_api, discriminator_simplified_api
from utils import get_image

# Defile TF Flags
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    # Print flags
    for flag, _ in FLAGS.__flags.items():
        print('"{}": {}'.format(flag, getattr(FLAGS, flag)))
    print("--------------------")

    # Configure checkpoint/samples dir
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    z_dim = 100 # noise dim

    # Construct graph on GPU
    with tf.device("/gpu:0"):

        """ Define Models """
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
        real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # Input noise into generator for training
        net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)

        # Input real and generated fake images into discriminator for training
        net_d, d_logits = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)
        net_d2, d2_logits = discriminator_simplified_api(real_images, is_train=True, reuse=True)

        # Input noise into generator for evaluation
        # set is_train to False so that BatchNormLayer behave differently
        net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

        """ Define Training Operations """
        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
        
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
        d_loss = d_loss_real + d_loss_fake

        # generator: try to make the the fake images look real (1)
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        # Define optimizers for updating discriminator and generator
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)

    # Init Session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)

    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))

    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)# sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    """ Training models """
    iter_counter = 0
    for epoch in range(FLAGS.epoch):

        # Shuffle data
        shuffle(data_files)

        # Update sample files based on shuffled data
        sample_files = data_files[0:FLAGS.sample_size]
        sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        print("[*] Sample images updated!")

        # Load image data
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

        for idx in range(0, batch_idxs):
            batch_files = data_files[idx*FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]

            # Get real images (more image augmentation functions at [http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html])
            batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
            start_time = time.time()
            
            # Updates the Discriminator(D)
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
            
            # Updates the Generator(G)
            # run generator twice to make sure that d_loss does not go to zero (different from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # Generate images
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                # Visualize generated images
                tl.visualize.save_images(img, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # Save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")
    
    sess.close()

if __name__ == '__main__':
    tf.app.run()
