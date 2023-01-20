import tensorlayerx as tlx
from tensorlayerx.nn import Linear, ConvTranspose2d, Reshape, BatchNorm2d, Conv2d, Flatten, Module

class Generator(Module):
    gf_dim = 64
    image_size = 64
    s16 = image_size // 16
    w_init = tlx.nn.initializers.random_normal(stddev=0.02)
    gamma_init = tlx.nn.initializers.random_normal(1., 0.02)

    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = Linear(out_features=self.gf_dim * 8 * self.s16 * self.s16,  W_init=self.w_init)
        self.reshape = Reshape(shape=(-1, self.s16, self.s16, self.gf_dim * 8))
        self.bn1 = BatchNorm2d(0.9, act=tlx.nn.ReLU, gamma_init=self.gamma_init)
        self.deconv2d1 = ConvTranspose2d(self.gf_dim * 4, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn2 = BatchNorm2d(0.9, act=tlx.nn.ReLU, gamma_init=self.gamma_init)
        self.deconv2d2 = ConvTranspose2d(self.gf_dim * 2, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn3 = BatchNorm2d(0.9, act=tlx.nn.ReLU, gamma_init=self.gamma_init)
        self.deconv2d3 = ConvTranspose2d(self.gf_dim, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn4 = BatchNorm2d(0.9, act=tlx.nn.ReLU, gamma_init=self.gamma_init)
        self.deconv2d4 = ConvTranspose2d(3, (5, 5), (2, 2), act=tlx.nn.ReLU, W_init=self.w_init)

    def forward(self, x):
        x = self.linear1(x)
        x = self.reshape(x)
        x = self.bn1(x)
        x = self.deconv2d1(x)
        x = self.bn2(x)
        x = self.deconv2d2(x)
        x = self.bn3(x)
        x = self.deconv2d3(x)
        x = self.bn4(x)
        x = self.deconv2d4(x)

        return x

class Discriminator(Module):

    df_dim = 64
    w_init = tlx.nn.initializers.random_normal(stddev=0.02)
    gamma_init = tlx.nn.initializers.random_normal(1., 0.02)

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2d(self.df_dim, (5, 5), (2, 2), act=tlx.nn.LeakyReLU, W_init=self.w_init)
        self.conv2 = Conv2d(self.df_dim * 2, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn1 = BatchNorm2d(0.9, act=tlx.nn.LeakyReLU, gamma_init=self.gamma_init)
        self.conv3 = Conv2d(self.df_dim * 4, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn2 = BatchNorm2d(0.9, act=tlx.nn.LeakyReLU, gamma_init=self.gamma_init)
        self.conv4 = Conv2d(self.df_dim * 8, (5, 5), (2, 2), W_init=self.w_init, b_init=None)
        self.bn3 = BatchNorm2d(0.9, act=tlx.nn.LeakyReLU, gamma_init=self.gamma_init)
        self.flatten = Flatten()
        self.linear = Linear(1, W_init=self.w_init)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x