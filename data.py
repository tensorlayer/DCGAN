import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision import transforms, load_images
## enable debug logging
tlx.logging.set_verbosity(tlx.logging.DEBUG)

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

tlx.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tlx.files.exists_or_mkdir(flags.sample_dir) # save generated image

transforms_celebA = transforms.Compose(
    [
        transforms.CentralCrop(size = [128, 128]),
        transforms.Resize(size=64),
        transforms.RandomFlipHorizontal(),
        transforms.Normalize(mean=(127.5), std=(127.5), data_format='HWC'),
    ]
)

class CELEBA(Dataset):

    def __init__(self):
        images_path = r'data/celebA/img_align_celeba'
        self.images = load_images(images_path, n_threads=0)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = transforms_celebA(image)
        return image

    def __len__(self):
        return len(self.images)


def get_celebA(batch_size):
    # dataset API and augmentation
    images_path = tlx.files.load_celebA_dataset()
    celebA = CELEBA()
    trainloader = DataLoader(celebA, batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader, images_path

