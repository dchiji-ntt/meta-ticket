# Based on https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/datasets/helpers.py

import warnings

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

# fixed version of torchmeta.transforms.augmentations
from utils.torchmeta_augmentations import Rotation
from utils.miniimagenet import MiniImagenet
from utils.cars import CARS
from utils.cub import CUB
from utils.aircraft import AirCraft
from utils.vgg_flower import VggFlower

from torchmeta.transforms import Categorical, ClassSplitter, SegmentationPairTransform
from torchmeta.datasets import (Omniglot, TieredImagenet, CIFARFS,
                                FC100, DoubleMNIST, TripleMNIST, Pascal5i)
from torchmeta.datasets.helpers import helper_with_default

class ToSingleChannel:
    """Rotate by one of the given angles."""

    def __init__(self):
        pass

    def __call__(self, x):
        ret = x.mean(dim=0)
        return ret

def omniglot_helper(folder, shots, ways, shuffle=True, test_shots=None,
             seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(28), ToTensor()]),
        'class_augmentations': [Rotation([90, 180, 270])]
    }

    return helper_with_default(Omniglot, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def doublemnist_helper(folder, shots, ways, shuffle=True, test_shots=None,
                seed=None, **kwargs):
    return helper_with_default(DoubleMNIST, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults={}, **kwargs)

def miniimagenet_helper(folder, shots, ways, shuffle=True, test_shots=None,
                        seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(84), ToTensor()]),
        'class_augmentations': []
    }

    return helper_with_default(MiniImagenet, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def miniimagenet_mono_helper(folder, shots, ways, shuffle=True, test_shots=None,
                        seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(28), ToTensor(), ToSingleChannel()]),
        'class_augmentations': []
    }

    return helper_with_default(MiniImagenet, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def tieredimagenet_helper(folder, shots, ways, shuffle=True, test_shots=None,
                          seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(84), ToTensor()]),
        'class_augmentations': []
    }

    return helper_with_default(TieredImagenet, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def cifarfs_helper(folder, shots, ways, shuffle=True, test_shots=None,
                   seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(32), ToTensor()])
    }
    return helper_with_default(CIFARFS, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)
def cifarfs_mono_helper(folder, shots, ways, shuffle=True, test_shots=None,
                   seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(28), ToTensor(), ToSingleChannel()]),
    }
    return helper_with_default(CIFARFS, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

# This is based on https://github.com/HJ-Yoo/BOIL/blob/master/torchmeta/datasets/helpers.py
def cars_helper(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(84), ToTensor()])
    }
    return helper_with_default(CARS, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def cub_helper(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, **kwargs):
    image_size = 84
    defaults = {
            'transform': Compose([
                Resize(int(image_size * 1.5)),
                CenterCrop(image_size),
                ToTensor()
                ])
            }
    return helper_with_default(CUB, folder, shots, ways,
                shuffle=shuffle, test_shots=test_shots,
                seed=seed, defaults=defaults, **kwargs)

def vgg_flower_helper(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(32), ToTensor()])
    }
    return helper_with_default(VggFlower, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

def aircraft_helper(folder, shots, ways, shuffle=True, test_shots=None,
                 seed=None, **kwargs):
    defaults = {
        'transform': Compose([Resize(32), ToTensor()])
    }
    return helper_with_default(AirCraft, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)

