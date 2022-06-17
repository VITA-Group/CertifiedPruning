from modules import Flatten
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable
# from advertorch.utils import NormalizeByChannelMeanStd


#__all__ = ['cnn_b_adv','cnn_7layer_bn','cifar_model','cifar_model_shallow','cifar_model_deep','cifar_model_ultra_deep']


def cnn_b_adv(num_classes=10,bn=True):
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10)
    )
    return model

def cnn_7layer_bn(num_classes=10,bn=True, in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,num_classes)
    )
    return model

def cifar_model(num_classes=10,bn=False):
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep(num_classes=10,bn=True,channels=None,greyscale=False):
    # cifar deep
    if channels==None:
        channels=[32,64,64,128,128]
    assert(len(channels)==5)
    module_list=[
        nn.Conv2d(1 if greyscale else 3, channels[0], 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels[3], channels[4], 4, stride=2, padding=1),
        nn.ReLU(),
        # nn.Conv2d(128, 256, 4, stride=2, padding=1),
        # nn.ReLU(),
        Flatten(),
        nn.Linear(channels[4]*((3*3) if greyscale else (4*4)), 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    ]

    new_module_list=[]
    for m in module_list:
        new_module_list.append(m)
        if bn and isinstance(m,nn.Conv2d):
            new_module_list.append(nn.BatchNorm2d(m.out_channels))

    model=nn.Sequential(*new_module_list)
    return model

