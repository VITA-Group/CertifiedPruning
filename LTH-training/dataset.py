'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''

import os 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100,SVHN,FashionMNIST
import torch

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders','svhn_dataloaders','FashionMNIST_dataloaders']

def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=4):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

#    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#    test_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_loader.mean = test_loader.mean = torch.tensor([0.4914, 0.4822, 0.4465])
    train_loader.std = test_loader.std = torch.tensor([0.2470, 0.2435, 0.2616])

    # return train_loader, val_loader, test_loader
    return train_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', num_workers=4):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673,0.2564,	0.2762])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673,0.2564,	0.2762])
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(50000)))
    # val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_loader.mean = test_loader.mean = torch.tensor([0.5071, 0.4866, 0.4409])
    train_loader.std = test_loader.std = torch.tensor([0.2673,0.2564,0.2762])


    return train_loader, test_loader


def FashionMNIST_dataloaders(batch_size=128, data_dir='datasets/FashionMNIST', num_workers=4):

    mean=[0.286]
    std=[0.315]
    # mean=[0]
    # std=[1]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = FashionMNIST(data_dir, train=True, transform=train_transform, download=True)
    test_set = FashionMNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_loader.mean = test_loader.mean = torch.tensor(mean)
    train_loader.std = test_loader.std = torch.tensor(std)

    return train_loader, test_loader


#based on https://github.com/KaidiXu/auto_LiRPA/blob/319252bbe6c51a60852687418575429caf9205ee/examples/vision/datasets.py#L107
def svhn_dataloaders(batch_size=128, data_dir='datasets/svhn', num_workers=4):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        transforms.Normalize(mean=[0,0,0], std=[1,1,1])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[1,1,1])
    #    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

#    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#    test_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    train_set = SVHN(data_dir, split='train', transform=train_transform, download=True)
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    train_loader.mean = test_loader.mean = torch.tensor([0,0,0])
    train_loader.std = test_loader.std = torch.tensor([1,1,1])

    # return train_loader, val_loader, test_loader
    return train_loader, test_loader
