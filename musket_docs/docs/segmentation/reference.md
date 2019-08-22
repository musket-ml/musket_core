# Segmentation specific root properties


## shape

shape of the input picture, in the form heigth,width, number of channels, all images will be resized to this shape before processing

Example:

```yaml
shape: (440,440,3)
```
 

## classes

Number of classes that should be segmented.

## activation

activation function that should be used in last layer. In the case of binary segmentation it usually should be `sigmoid` if you have
more then one class than most likely you need to use `softmax`, but actually you are free to use any activation function that is
registered in Keras


## architecture

This property configures decoder architecture that should be used:

At this moment segmentation pipeline supports following architectures:

- [Unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Linknet](https://codeac29.github.io/projects/linknet/)
- [PSP](https://arxiv.org/abs/1612.01105)
- [FPN](https://arxiv.org/abs/1612.03144)
- [DeeplabV3](https://arxiv.org/abs/1706.05587)

  
## backbone

This property configures encoder that should be used:


`FPN`, `PSP`, `Linkenet`, `UNet` architectures support following backbones: 

  - [VGGNet](https://arxiv.org/abs/1409.1556)
    - vgg16
    - vgg19
  - [ResNet](https://arxiv.org/abs/1512.03385)
    - resnet18
    - resnet34
    - resnet50 
    - resnet101
    - resnet152
  - [ResNext](https://arxiv.org/abs/1611.05431)
    - resnext50
    - resnext101
  - [DenseNet](https://arxiv.org/abs/1608.06993)
    - densenet121
    - densenet169
    - densenet201
  - [Inception-v3](https://arxiv.org/abs/1512.00567)
  - [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261)
  
All them support the weights pretrained on [ImageNet](http://www.image-net.org/):
```yaml
encoder_weights: imagenet
```

At this moment `DeeplabV3` architecture supports following backbones:
 - [MobileNetV2](https://arxiv.org/abs/1801.04381)
 - [Xception](https://arxiv.org/abs/1610.02357)

Deeplab supports weights pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/):

## encoder_weights
  
This property configures initial weights of the encoder, supported values:

`imagenet`  

## encoder_weights
  
This property configures initial weights of the encoder, supported values:

`imagenet`

## testTimeAugmentation

This property turns on test time augmentation mechanism 