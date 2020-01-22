'''
Created on 9 Jan 2020

@author: 32kda
'''
from musket_core import configloader
from imgaug.augmenters import Augmenter

def augmenter(aug_class):
    configloader.load("augmenters").register_member(aug_class.__name__, aug_class)
    if not issubclass(aug_class, Augmenter):
        raise ValueError('{} is not an instance of imgaug.augmenters.Augmenter!'.format(aug_class.__name__))
    return aug_class 
