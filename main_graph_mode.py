"""Graph mode, single GPU
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

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(save_dir)

    n_step_epoch = int(len(images_path) // FLAGS.batch_size)
    # print(n_step_epoch)
    # v = sess.run(images)
    # print(v.shape, v.max(), v.min())
    # tl.vis.save_images(v, [8, 8])
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

    #
    # ## old implementation
    #
    # data_files = np.array(glob(os.path.join("./data", FLAGS.dataset, "*.jpg")))
    # num_files = len(data_files)
    #
    # # Mini-batch generator
    # def iterate_minibatches(batch_size, shuffle=True):
    #     if shuffle:
    #         indices = np.random.permutation(num_files)
    #     for start_idx in range(0, num_files - batch_size + 1, batch_size):
    #         if shuffle:
    #             excerpt = indices[start_idx: start_idx + batch_size]
    #         else:
    #             excerpt = slice(start_idx, start_idx + batch_size)
    #         # Get real images (more image augmentation functions at [http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html])
    #         yield np.array([get_image(file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0)
    #                         for file in data_files[excerpt]]).astype(np.float32)
    #
    # batch_steps = min(num_files, FLAGS.train_size) // FLAGS.batch_size
    #
    # # sample noise
    # sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, FLAGS.z_dim)).astype(np.float32)
    #
    # """ Training models """
    # iter_counter = 0
    # for epoch in range(FLAGS.n_epoch):
    #
    #     sample_images = next(iterate_minibatches(FLAGS.sample_size))
    #     print("[*] Sample images updated!")
    #
    #     steps = 0
    #     for batch_images in iterate_minibatches(FLAGS.batch_size):
    #
    #         batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.batch_size, FLAGSz_dim)).astype(np.float32)
    #         start_time = time.time()
    #
    #         # Updates the Discriminator(D)
    #         errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images})
    #
    #         # Updates the Generator(G)
    #         # run generator twice to make sure that d_loss does not go to zero (different from paper)
    #         for _ in range(2):
    #             errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
    #
    #         end_time = time.time() - start_time
    #         print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
    #                 % (epoch, FLAGS.n_epoch, steps, batch_steps, end_time, errD, errG))
    #
    #         iter_counter += 1
    #         if np.mod(iter_counter, FLAGS.sample_step) == 0:
    #             # Generate images
    #             img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z: sample_seed, real_images: sample_images})
    #             # Visualize generated images
    #             tl.visualize.save_images(img, [num_tiles, num_tiles], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, steps))
    #             print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
    #
    #         if np.mod(iter_counter, FLAGS.save_step) == 0:
    #             # Save current network parameters
    #             print("[*] Saving checkpoints...")
    #             tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
    #             tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
    #             print("[*] Saving checkpoints SUCCESS!")
    #
    #         steps += 1

    sess.close()

if __name__ == '__main__':
    train()
    # try:
    #     tf.app.run()
    # except KeyboardInterrupt:
    #     print('EXIT')
