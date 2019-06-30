# DCGAN in TensorLayer


This is the TensorLayer implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).
Looking for Text to Image Synthesis ? [click here](https://github.com/zsdonghao/text-to-image)

![alt tag](img/DCGAN.png)


- 🆕 🔥 2019 May: We just update this project to support TF2 and TL2. Enjoy!
- 🆕 🔥 2019 May: This project is chosen as the default template of TL projects.


## Prerequisites

- Python3.5 3.6
- TensorFlow==2.0.0a0  `pip3 install tensorflow-gpu==2.0.0a0`
- TensorLayer=2.1.0		`pip3 install tensorlayer==2.1.0`

## Usage

First, download the aligned face images from [google](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8) or [baidu](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F) to a `data` folder.

Second, train the GAN:

    $ python train.py
    
## Result on celebA


<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="90%" height="90%"/>
</div>
</a>
