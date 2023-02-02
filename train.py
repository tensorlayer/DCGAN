import os
# os.environ['TL_BACKEND'] = 'tensorflow' # Just modify this line, easily switch to any framework!
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'torch'
import time
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.utils.visualize import save_images
from tensorlayerx.model import TrainOneStep
from data import get_celebA, flags
from model import  Generator, Discriminator
# tlx.set_device('GPU') # use this ops set default device.
num_tiles = int(np.sqrt(flags.sample_size))

class WithLoss_D(Module):
    def __init__(self, D, G):
        super(WithLoss_D, self).__init__()
        self.D = D
        self.G = G

    def forward(self, images, fake):
        d_logits = self.D(self.G(fake))
        d2_logits = self.D(images)
        d_loss_real = tlx.losses.sigmoid_cross_entropy(d2_logits, tlx.ones_like(d2_logits))
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tlx.losses.sigmoid_cross_entropy(d_logits, tlx.zeros_like(d_logits))
        d_loss = d_loss_real + d_loss_fake
        return d_loss

class WithLoss_G(Module):
    def __init__(self, D, G):
        super(WithLoss_G, self).__init__()
        self.D = D
        self.G = G

    def forward(self, images, fake):
        d_logits = self.D(self.G(fake))
        g_loss = tlx.losses.sigmoid_cross_entropy(d_logits, tlx.ones_like(d_logits))
        return g_loss

def train():
    images_loader, images_path = get_celebA(flags.batch_size)
    G = Generator()
    D = Discriminator()
    G.init_build(tlx.nn.Input(shape=(flags.batch_size, 100)))
    D.init_build(tlx.nn.Input(shape=(flags.batch_size, 64, 64, 3)))

    G.set_train()
    D.set_train()

    d_optimizer = tlx.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tlx.optimizers.Adam(flags.lr, beta_1=flags.beta1)

    g_weights = G.trainable_weights
    d_weights = D.trainable_weights


    net_with_loss_D = WithLoss_D(D, G)
    net_with_loss_G = WithLoss_G(D, G)
    trainforG = TrainOneStep(net_with_loss_G, optimizer=g_optimizer, train_weights=g_weights)
    trainforD = TrainOneStep(net_with_loss_D, optimizer=d_optimizer, train_weights=d_weights)

    n_step_epoch = int(len(images_path) // flags.batch_size)
    
    # Z = tf.distributions.Normal(0., 1.)
    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images_loader):
            step_time = time.time()
            z = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
            z = tlx.ops.convert_to_tensor(z)
            d_loss = trainforD(batch_images, z)
            g_loss = trainforG(batch_images, z)

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch, \
                  flags.n_epoch, step, n_step_epoch, time.time()-step_time, float(d_loss), float(g_loss)))

        if np.mod(epoch, flags.save_every_epoch) == 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            G.set_eval()
            result = G(z)
            G.set_train()
            save_images(tlx.convert_to_numpy(result), [num_tiles, num_tiles], '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))

if __name__ == '__main__':
    train()
