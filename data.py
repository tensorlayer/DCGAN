import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)

class FLAGS(object):
    def __init__(self):
        self.n_epoch = 25 # "Epoch to train [25]"
        self.z_dim = 100 # "Num of noise value]"
        self.lr = 0.0002 # "Learning rate of for adam [0.0002]")
        self.beta1 = 0.5 # "Momentum term of adam [0.5]")
        self.batch_size = 64 # "The number of batch images [64]")
        self.output_size = 64 # "The size of the output images to produce [64]")
        self.sample_size = 64 # "The number of sample images [64]")
        self.c_dim = 3 # "Number of image channels. [3]")
        self.save_every_epoch = 1 # "The interval of saveing checkpoints.")
        # self.dataset = "celebA" # "The name of dataset [celebA, mnist, lsun]")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        assert np.sqrt(self.sample_size) % 1 == 0., 'Flag `sample_size` needs to be a perfect square'
flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image

def get_celebA(output_size, n_epoch, batch_size):
    # dataset API and augmentation
    images_path = tl.files.load_file_list(path='data', regx='.*.jpg', keep_prefix=True, printable=False)
    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')
    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.crop_central(image, [FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
        # image = tf.image.resize_images(image, FLAGS.output_size])
        image = image[45:173, 25:153, :] # central crop
        image = tf.image.resize([image], (output_size, output_size))[0]
        # image = tf.image.crop_and_resize(image, boxes=[[]], crop_size=[64, 64])
        # image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.output_size, FLAGS.output_size) # central crop
        image = tf.image.random_flip_left_right(image)
        image = image * 2 - 1
        return image
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    # ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, images_path
    # for batch_images in train_ds:
    #     print(batch_images.shape)
    # value = ds.make_one_shot_iterator().get_next()
