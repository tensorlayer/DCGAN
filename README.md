# DCGAN in TensorLayer

TensorLayer implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).

Looking for Text to Image Synthesis ? [click here](https://github.com/zsdonghao/text-to-image)

![alt tag](img/DCGAN.png)

## Prerequisites

- Python3
- TensorFlow==1.13
- TensorLayer (self-contained)

## Usage

First, download the aligned face images from [google](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8) or [baidu](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F) to a `data` folder.

Second, train the GAN:

    $ python main_eager_mode.py   # single GPU for TF>=1.13
    $ python main_graph_mode.py   # single GPU for TF<=1.13
    $ python main_eager_mode_horovod.py   # multiple GPU (TODO)
    $ python main_eager_mode_tlmagic.py   # multiple GPU (TODO)    

## Result on celebA


<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="90%" height="90%"/>
</div>
</a>