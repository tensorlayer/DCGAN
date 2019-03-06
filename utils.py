import tensorflow as tf
import tensorlayer as tl

def get_celebA(output_size, n_epoch, batch_size):
    # dataset API and augmentation
    images_path = tl.files.load_file_list(path='data', regx='.*.jpg', keep_prefix=True, printable=False)
    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')
    def _map_fn(image_path):
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.crop_central(image, [FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
        # image = tf.image.resize_images(image, FLAGS.output_size])
        image = image[45:173, 25:153, :]
        image = tf.image.resize_bicubic([image], (output_size, output_size))[0]
        # image = tf.image.crop_and_resize(image, boxes=[[]], crop_size=[64, 64])
        # image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.output_size, FLAGS.output_size) # central crop
        image = tf.image.random_flip_left_right(image)
        image = image * 2 - 1
        return image
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, images_path
    # for batch_images in train_ds:
    #     print(batch_images.shape)
    # value = ds.make_one_shot_iterator().get_next()


## old code
# import scipy.misc
# import imageio as io
# import numpy as np
#
# def center_crop(x, crop_h, crop_w=None, resize_w=64):
#     if crop_w is None:
#         crop_w = crop_h
#     h, w = x.shape[:2]
#     j = int(round((h - crop_h)/2.))
#     i = int(round((w - crop_w)/2.))
#     return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
#                                [resize_w, resize_w])
#
# def merge(images, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[j * h: j * h + h, i * w: i * w + w, :] = image
#     return img
#
# def transform(image, npx=64, is_crop=True, resize_w=64):
#     if is_crop:
#         cropped_image = center_crop(image, npx, resize_w=resize_w)
#     else:
#         cropped_image = image
#     return (np.array(cropped_image) / 127.5) - 1.
#
# def inverse_transform(images):
#     return (images + 1.) / 2.
#
# def imread(path, is_grayscale = False):
#     if (is_grayscale):
#         return io.imread(path).astype(np.float).flatten()
#     else:
#         return io.imread(path).astype(np.float)
#
# def imsave(images, size, path):
#     return io.imsave(path, merge(images, size))
#
# def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
#     return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)
#
# def save_images(images, size, image_path):
#     return imsave(inverse_transform(images), size, image_path)
