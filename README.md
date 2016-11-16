# DCGAN in TensorFlow

TensorFlow / TensorLayer implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks.

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*


## Prerequisites

- Python 2.7 or Python 3.3+
- [TensorFlow==0.10.0 or higher](https://www.tensorflow.org/)
- [TensorLayer==1.2.6 or higher](https://github.com/zsdonghao/tensorlayer) (already in this repo)


## Usage

First, download dataset with:

    $ python download.py celebA		[202599 face images]

To train a model with downloaded dataset:

    $ python main.py


