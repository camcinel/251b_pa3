# PASCAL VOC 2007 Image Segmentation

## Description

This python module trains a few different neural networks to perform image segmentation on the [Pascal VOC 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).
This module can run on CPU or GPU if CUDA is enabled.
Currently four different models are implemented:

1. A simple fully convolution network
2. A pretrained [ResNet34](https://arxiv.org/abs/1512.03385) encoder with a convolutional decoder
3. [UNet](https://arxiv.org/abs/1505.04597)
4. A custom convolutional neural network

Additionally, the following loss functions are implemented:

1. Cross entropy, weighted and unweighted
2. Dice Loss, weighted and unweighted
3. Focal Loss, weighted and unweighted

## Usage

To use the model, the data must first be downloaded by running the script
```
python download.py
```

The model can be trained and tested via
```
python train.py
```

### Command Line Arguments

The following command line arguments are supported:

- `give-time`: will give the time elapsed for each epoch of training.
The default is not to give time.

## Required Libraries

This module requires an installation of [Pytorch](https://pytorch.org/get-started/locally/) as well as CUDA if GPU training is desired.
The following additional libraries are required:

- `numpy`
- `matplotlib`
- `pillow`

## File Structure

The module is broken up into the following files:

- `download.py`: script to download the PASCAL VOC 2007 dataset
- `images.py`: methods to visualize dataset and output images
- `voc.py`: methods to load the VOC dataset
- `util.py`: various helper methods and classes, including plotting and loss functions
- `basic_fcn.py`: contains the constructor for the simple fully convolutional network
- `resnet.py`: contains the constructor for the pretrained ResNet34
- `unet.py`: contains the constructor for UNet
- `train.py`: main file, runnings the training loop