# PASCAL VOC 2007 Image Segmentation

## Description

This python module trains a few different neural networks to perform image segmentation on the [Pascal VOC 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).
This module can run on CPU or GPU if CUDA is enabled.
Currently four different models are implemented:

1. A simple fully convolution network
2. A pretrained [ResNet34](https://arxiv.org/abs/1512.03385) encoder with a convolutional decoder
3. [UNet](https://arxiv.org/abs/1505.04597)
4. A custom convolutional neural network based on [FCN8](https://arxiv.org/abs/1411.4038)

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

- `--time`: will give the time elapsed for each epoch of training. The default is not to give time.
- `--batch-size`: determines the number of images in each batch. Default is 16.
- `--epochs`: the maximum number of training of epochs. Default is 100.
- `--learning-rate`: the learning rate during training. Default if 0.005.
- `--model`: chooses the model to train and test. The choices are
    1. `baseline`: the baseline model
    2. `resnet`: a pre-trained ResNet34 encoder and custom decoder
    3. `unet`: an implementation of UNet
    4. `fc8`: a custom convolutional network based on FCN8
- `--weighted`: weight loss function by inverse class frequency
- `--loss`: chooses the model's loss function. The choices are
  1. `cross-entropy`: cross entropy loss
  2. `focal`: focal loss
  3. `dice`: dice loss
- `--patience`: chooses the amount of patience epochs for early stopping. The default is 10
- `--scheduler`: chooses to use a cosine annealing learning rate scheduler. The default is not to use cosine annealing.

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
- `fc8.py`: contains the constructor for custom network based on FCN8
- `train.py`: main file, runs the training loop