from models.ResNet import *
from models.ResNets import *
from models.VGG import * 
from models.FeedforwardNets import * 

model_dict = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet20s': resnet20s,
    'resnet44s': resnet44s,
    'resnet56s': resnet56s,
    'vgg16_bn': vgg16_bn,
    'cifar_model_deep': cifar_model_deep,
    'cifar_model': cifar_model,
    'cnn_7layer_bn': cnn_7layer_bn,
    'cnn_b_adv': cnn_b_adv,
}

