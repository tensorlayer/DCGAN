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

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate face image.

Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
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

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)



def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(s2, s2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits


def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits



def generator(inputs, is_train=True, reuse=False):
    output_size = 64
    s = output_size
    s2, s4, s8, s16 = int(output_size/2), int(output_size/4), int(output_size/8), int(output_size/16)
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color
    batch_size = FLAGS.batch_size

    if reuse:
        tf.get_variable_scope().reuse_variables()
        tl.layers.set_name_reuse(reuse)

    net_in = tl.layers.InputLayer(inputs, name='g/in')
    net_h0 = tl.layers.DenseLayer(net_in, n_units = gf_dim*8*s16*s16,
                                W_init = tf.random_normal_initializer(stddev=0.02),
                                act = tf.identity, name='g/h0/lin')
    # print(net_h0.outputs)     # (64, 8192)
    net_h0 = tl.layers.ReshapeLayer(net_h0, shape = [-1, s16, s16, gf_dim*8], name='g/h0/reshape')
    net_h0 = tl.layers.BatchNormLayer(net_h0, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/h0/batch_norm')
    net_h0.outputs = tf.nn.relu(net_h0.outputs, name='g/h0/relu')
    # print(net_h0.outputs)     # (64, 4, 4, 512)

    net_h1 = tl.layers.DeConv2dLayer(net_h0,
                                shape = [5, 5, gf_dim*4, gf_dim*8],
                                output_shape = [batch_size, s8, s8, gf_dim*4],
                                strides=[1, 2, 2, 1],
                                W_init = tf.random_normal_initializer(stddev=0.02),
                                act=tf.identity, name='g/h1/decon2d')
    net_h1 = tl.layers.BatchNormLayer(net_h1, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/h1/batch_norm')
    net_h1.outputs = tf.nn.relu(net_h1.outputs, name='g/h1/relu')
    # print(net_h1.outputs)     # (64, 8, 8, 256)

    net_h2 = tl.layers.DeConv2dLayer(net_h1,
                                shape = [5, 5, gf_dim*2, gf_dim*4],
                                output_shape = [batch_size, s4, s4, gf_dim*2],
                                strides=[1, 2, 2, 1],
                                W_init = tf.random_normal_initializer(stddev=0.02),
                                act=tf.identity, name='g/h2/decon2d')
    net_h2 = tl.layers.BatchNormLayer(net_h2, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/h2/batch_norm')
    net_h2.outputs = tf.nn.relu(net_h2.outputs, name='g/h2/relu')
    # print(net_h2.outputs)     # (64, 16, 16, 128)

    net_h3 = tl.layers.DeConv2dLayer(net_h2,
                                shape = [5, 5, gf_dim*1, gf_dim*2],
                                output_shape = [batch_size, s2, s2, gf_dim*1],
                                strides=[1, 2, 2, 1],
                                W_init = tf.random_normal_initializer(stddev=0.02),
                                act=tf.identity, name='g/h3/decon2d')
    net_h3 = tl.layers.BatchNormLayer(net_h3, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/h3/batch_norm')
    net_h3.outputs = tf.nn.relu(net_h3.outputs, name='g/h3/relu')
    # print(net_h3.outputs)     # (64, 32, 32, 64)
    net_h4 = tl.layers.DeConv2dLayer(net_h3,
                                shape = [5, 5, c_dim, gf_dim*1],
                                output_shape = [batch_size, output_size, output_size, c_dim],
                                strides=[1, 2, 2, 1],
                                W_init = tf.random_normal_initializer(stddev=0.02),
                                act=tf.identity, name='g/h4/decon2d')
    # print(net_h3.outputs)     # (64, 64, 64, 3)
    logits = net_h4.outputs
    net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits


def discriminator(inputs, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color
    batch_size = FLAGS.batch_size

    if reuse:
        tf.get_variable_scope().reuse_variables()
        tl.layers.set_name_reuse(reuse)

    net_in = tl.layers.InputLayer(inputs, name='d/in')

    net_h0 = tl.layers.Conv2dLayer(net_in, shape=[5, 5, c_dim, df_dim],
                                   W_init = tf.random_normal_initializer(stddev=0.02),
                                   strides=[1, 2, 2, 1], name='d/h0/conv2d')
    net_h0.outputs = tl.activation.leaky_relu(net_h0.outputs, alpha=0.2, name='d/h0/lrelu')
    # print(net_h0.outputs)   # (64, 32, 32, 64)
    net_h1 = tl.layers.Conv2dLayer(net_h0, shape=[5, 5, df_dim, df_dim*2],
                                   W_init = tf.random_normal_initializer(stddev=0.02),
                                   strides=[1, 2, 2, 1], name='d/h1/conv2d')
    net_h1 = tl.layers.BatchNormLayer(net_h1, is_train=is_train, name='d/h1/batch_norm')
    net_h1.outputs = tl.activation.leaky_relu(net_h1.outputs, alpha=0.2, name='d/h1/lrelu')
    # print(net_h1.outputs)   # (64, 16, 16, 128)
    net_h2 = tl.layers.Conv2dLayer(net_h1, shape=[5, 5, df_dim*2, df_dim*4],
                                   W_init = tf.random_normal_initializer(stddev=0.02),
                                   strides=[1, 2, 2, 1], name='d/h2/conv2d')
    net_h2 = tl.layers.BatchNormLayer(net_h2, is_train=is_train, name='d/h2/batch_norm')
    net_h2.outputs = tl.activation.leaky_relu(net_h2.outputs, alpha=0.2, name='d/h2/lrelu')
    # print(net_h2.outputs)   # (64, 8, 8, 256)
    net_h3 = tl.layers.Conv2dLayer(net_h2, shape=[5, 5, df_dim*4, df_dim*8],
                                   W_init = tf.random_normal_initializer(stddev=0.02),
                                   strides=[1, 2, 2, 1], name='d/h3/conv2d')
    net_h3 = tl.layers.BatchNormLayer(net_h3, is_train=is_train, name='d/h3/batch_norm')
    net_h3.outputs = tl.activation.leaky_relu(net_h3.outputs, alpha=0.2, name='d/h3/lrelu')
    # print(net_h3.outputs)   # (64, 4, 4, 512)
    net_h4 = tl.layers.FlattenLayer(net_h3, name='d/h4/flatten')
    net_h4 = tl.layers.DenseLayer(net_h4, n_units=1, act=tf.identity,
                                    W_init = tf.random_normal_initializer(stddev=0.02),
                                    name='d/h4/lin_sigmoid')
    # print(net_h4.outputs)   # (64, 1)
    logits = net_h4.outputs
    net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits



def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    z_dim = 100

    # with tf.device("/gpu:0"): # <-- if you have a GPU machine
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

    # z --> generator for training
    net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)
    # generated fake images --> discriminator
    net_d, d_logits = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)
    # real images --> discriminator
    net_d2, d2_logits = discriminator_simplified_api(real_images, is_train=True, reuse=True)
    # sample_z --> generator for evaluation, set is_train to False
    # so that BatchNormLayer behave differently
    net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d2_logits, tf.ones_like(d2_logits)))    # real == 1
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits, tf.zeros_like(d_logits)))     # fake == 0
    d_loss = d_loss_real + d_loss_fake
    # generator: try to make the the fake images look real (1)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits, tf.ones_like(d_logits)))

    # trainable parameters for updating discriminator and generator
    g_vars = net_g.all_params   # only updates the generator
    d_vars = net_d.all_params   # only updates the discriminator

    net_g.print_params(False)
    print("---------------")
    net_d.print_params(False)

    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars)

    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
    sess.run(tf.initialize_all_variables())

    # load checkpoints
    print("[*] Loading checkpoints...")
    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')
    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading checkpoints failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        print("[*] Loading checkpoints SUCCESS!")


    # TODO: use minbatch to shuffle and iterate
    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))


    # TODO: shuffle sample_files each epoch
    sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    iter_counter = 0
    for epoch in range(FLAGS.epoch):
        #shuffle data
        shuffle(data_files)
        print("[*]Dataset shuffled!")

        # update sample files based on shuffled data
        sample_files = data_files[0:FLAGS.sample_size]
        sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        print("[*]Sample images updated!")

        # load image data
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            # get real images
            batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            start_time = time.time()
            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs,
                        time.time() - start_time, errD, errG))
            sys.stdout.flush()

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # generate and visualize generated images
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                '''
                img255 = (np.array(img) + 1) / 2 * 255
                tl.visualize.images2d(images=img255, second=0, saveable=True,
                                name='./{}/train_{:02d}_{:04d}'.format(FLAGS.sample_dir, epoch, idx), dtype=None, fig_idx=2838)
                '''
                save_images(img, [8, 8],
                            './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
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


if __name__ == '__main__':
    tf.app.run()
