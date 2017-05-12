import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate face image.

Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
#flags.DEFINE_integer("image_size", 28, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("image_size", 500, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 2, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "loam", "The name of dataset [celebA, mnist, loam, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def log(x):
    return tf.log(x + 1e-8)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)

    z_dim = 512

    ##========================= DEFINE MODEL ===========================##
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')
    
    # Generator: z --> hat_X
    net_g, g_logits = decoder_simplified_api(z, is_train=True, reuse=False)
    # Encoder:   X --> hat_z
    net_e, e_logits = encoder_simplified_api(real_images, is_train=True, reuse=False)

    # Discriminator for (hat_X, z)
    net_d, d_logits = discriminator_ali_api(net_g.outputs, z, is_train=True, reuse=False)
    # Discriminator for (X, hat_z)
    net_d2, d2_logits = discriminator_ali_api(real_images, net_e.outputs, is_train=True, reuse=True)

    with tf.name_scope('fake_image'):
        fake_image = tf.reshape(net_g.outputs, [-1, 64, 64, 3])
        tf.summary.image('fake', fake_image, 64)
        
    ##========================= DEFINE TRAIN OPS =======================##
    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1

    # Apply Different Loss
    with tf.name_scope('discriminator'):

        '''
        # Original Loss
        d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
        tf.summary.scalar('d_loss_real', d_loss_real)
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss', d_loss)
        # generator: try to make the the fake images look real (1)
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')
        tf.summary.scalar('g_loss', g_loss)
        '''

        """ Least Square Loss """
        d_loss = -tf.reduce_mean(log(d2_logits) + log(1-d_logits))
        g_loss = -tf.reduce_mean(log(d_logits) + log(1-d2_logits))
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)

        """ Least Square Loss """
        #d_loss = 0.5 * (tf.reduce_mean((d2_logits - 1)**2) + tf.reduce_mean((d_logits)**2))
        #tf.summary.scalar('d_loss', d_loss)
        # generator: try to make the the fake images look real (1)
        #g_loss = 0.5 * tf.reduce_mean((d_logits - 1)**2)
        #tf.summary.scalar('g_loss', g_loss)
        

        """ W-GAN """
        #d_loss = tf.reduce_mean(d2_logits) - tf.reduce_mean(d_logits)
        #g_loss = -tf.reduce_mean()
        # D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
        #             .minimize(-D_loss, var_list=theta_D))
        #G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
        #            .minimize(G_loss, var_list=theta_G))        
        #clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]



        """ Total Variation """
        # D_loss = -(tf.reduce_mean(0.5 * tf.nn.tanh(D_real)) -
        #            tf.reduce_mean(0.5 * tf.nn.tanh(D_fake)))
        # G_loss = -tf.reduce_mean(0.5 * tf.nn.tanh(D_fake))
        
        """ Forward KL """
        # D_loss = -(tf.reduce_mean(D_real) - tf.reduce_mean(tf.exp(D_fake - 1)))
        # G_loss = -tf.reduce_mean(tf.exp(D_fake - 1))
        
        """ Reverse KL """
        # D_loss = -(tf.reduce_mean(-tf.exp(D_real)) - tf.reduce_mean(-1 - D_fake))
        # G_loss = -tf.reduce_mean(-1 - D_fake)
        
        """ Pearson Chi-squared """
        #d_loss = -(tf.reduce_mean(d2_logits-1) - tf.reduce_mean(0.25*d_logits**2 + d_logits))
        #tf.summary.scalar('d_loss', d_loss)
        #g_loss = -tf.reduce_mean(0.25*(d_logits-1)**2 + (d_logits-1))
        #tf.summary.scalar('g_loss', g_loss)

        """ Squared Hellinger """
        # D_loss = -(tf.reduce_mean(1 - tf.exp(D_real)) -
        #            tf.reduce_mean((1 - tf.exp(D_fake)) / (tf.exp(D_fake))))
        # G_loss = -tf.reduce_mean((1 - tf.exp(D_fake)) / (tf.exp(D_fake)))

    
    with tf.name_scope('generator'):
        g_vars = tl.layers.get_variables_with_name('ENCODER', True, True)
        g2_vars = tl.layers.get_variables_with_name('DECODER', True, True)
        g_vars.extend(g2_vars)
        #variable_summaries(g_vars)

    with tf.name_scope('discriminator'):
        d_vars = tl.layers.get_variables_with_name('DISCRIMINATOR', True, True)
        #variable_summaries(d_vars)

    net_g.print_params(False)
    print("---------------")
    net_d.print_params(False)
    
    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars)

    # Limit GPU usage
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    tl.layers.initialize_global_variables(sess)

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
    # sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    merged = tf.summary.merge_all()
    logger = tf.summary.FileWriter('/tmp/tensorflow/ali/', sess.graph)
    tf.global_variables_initializer().run()

    # dataset: 0 for loam, 1 for mnist
    dataset = 0
    ##========================= TRAIN MODELS ================================##
    iter_counter = 0
    for epoch in range(FLAGS.epoch):
        ## shuffle data
        shuffle(data_files)
        print("[*] Dataset shuffled!")

        ## update sample files based on shuffled data
        sample_files = data_files[0:FLAGS.sample_size]
        sample = [get_image(sample_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        print("[*] Sample images updated!")
        print sample_images.shape

        ## load image data
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

        for idx in xrange(0, batch_idxs):
            ### Get datas ###
            batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            ## get real images
            # more image augmentation functions in http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html
            batch = [get_image(batch_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            # batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
            start_time = time.time()

            ### Update Nets ###
            # updates the discriminator
            feed_dict={z: batch_z, real_images: batch_images }
            errD, _ = sess.run([d_loss, d_optim], feed_dict=feed_dict)
            errG, _ = sess.run([g_loss, g_optim], feed_dict=feed_dict)
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, FLAGS.epoch, idx, batch_idxs,
                     time.time() - start_time, errD, errG))
            sys.stdout.flush()

            '''
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, FLAGS.epoch, idx, batch_idxs,
                         time.time() - start_time, errD, errG))
                sys.stdout.flush()
            '''
            iter_counter += 1
            print iter_counter
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # generate and visualize generated images
                summary, img, errD, errG = sess.run([merged, net_g.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                logger.add_summary(summary, iter_counter)
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
                sys.stdout.flush()

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
                save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    # the latest version location
                    net_g_name = os.path.join(save_dir, 'net_g.npz')
                    net_d_name = os.path.join(save_dir, 'net_d.npz')
                    # this version is for future re-check and visualization analysis
                    net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)
                    net_d_iter_name = os.path.join(save_dir, 'net_d_%d.npz' % iter_counter)
                    tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                    tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                    tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
                    tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)
                    print("[*] Saving checkpoints SUCCESS!")
                    logger.close()

if __name__ == '__main__':
    tf.app.run()
