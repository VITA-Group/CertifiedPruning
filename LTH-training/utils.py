'''
    setup model and datasets
'''

import logging
import os
# import sys
# import copy
# import torch
# import numpy as np
import math
import yaml
import argparse
from fractions import Fraction

from models import *
from dataset import *
from auto_LiRPA.utils import logger
from auto_LiRPA.bound_ops import BoundBatchNormalization
from collections import OrderedDict

def set_env(parser):
    add_common_arguments(parser)
    args = parser.parse_args()
    parse_config_file(args)
    parser.parse_args(namespace=args)  # parse again to overwrite config from command line
    args.eps = float(Fraction(args.eps))
    args.save_dir = os.path.join(os.environ['HOME'], 'saved_models', args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = args.save_dir + '/train.log'
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    logger.handlers = [logger.handlers[0]]  # ad-hoc fix bug
    return args

def add_common_arguments(parser):
    ##################################### Dataset #################################################
    parser.add_argument('--data', type=str, default='~/data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

    ##################################### Architecture ############################################
    parser.add_argument('--arch', type=str, default='cifar_model_deep', help='model architecture')
    parser.add_argument('--bn', default=1, type=int, help='whether to use batchNorm for FeedforwardNets')
    parser.add_argument('--seed', default=100, type=int, help='random seed')

    ##################################### General setting ############################################
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--resume_state', type=int, default=0, help="resume pruning state")
    parser.add_argument('--save_dir', help='The directory used to save the trained models', default='default', type=str)


    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--clip', type=float, default=8, help='gradient clip')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument("--lr_decay_milestones", nargs='+', type=int, default=[140, 170], help='learning rate dacay milestones')

    ##################################### Robustness setting #################################################
    parser.add_argument('--eps', default='2/255', type=str, help='pertubation amplitude')
    parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                        choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler",'FixedScheduler'], help='epsilon scheduler')
    parser.add_argument("--scheduler_opts", type=str, default="start=11,length=81,mid=0.4",
                        help='options for epsilon scheduler')

    ##################################### Pruning setting #################################################
    parser.add_argument('--rewind_epoch', default=8, type=int, help='rewind checkpoint')
    parser.add_argument('--prune_type', default='lt', type=str, choices=['lt', 'pt', 'rewind_lt', 'finetune'])
    parser.add_argument('--prune_times', default=16, type=int, help='overall times of pruning')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
    parser.add_argument('--layerwise', default=0, type=int, help='whether to prune layerwise in structured setting')
    parser.add_argument('--prune_method', default='nrs', choices=['random','l1unstruct','snip','taylor1ScorerAbs',
                                                                  'l1-channel','slim','nrs','hydra','refill'],
                        help='structured pruning method: random / l1unstruct / slim / nrs / l1-channel / hydra')
    parser.add_argument('--one_shot',default=0, type=int, help='whether to use one-shot pruning')
    parser.add_argument('--slim_weight', default=0, type=float, help='weight of slim loss')
    parser.add_argument('--rs_weight', default=0.01, type=float, help='weight of rs loss')
    parser.add_argument('--rs_reg', default=0,type=int, help='whether to use nrs loss as regularizer')
    parser.add_argument('--gamma_pow', default=2,type=int)


def parse_config_file(args):
    yaml_txt = open(args.config).read()
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    args.__dict__.update(loaded_yaml)
    for x in args.__dict__.keys():
        if type(args.__dict__[x])==dict:
            args.__dict__[x]=dict_to_namespace(args.__dict__[x])

def dict_to_namespace(args):
    for key in args.keys():
        if type(args[key]) == dict:
            args[key] = dict_to_namespace(args[key])
    ns = argparse.Namespace()
    ns.__dict__.update(args)

    return ns


def transform_pertub(dataset,norm_input,pertub,bound=False):
    #if use bound=True, we assume pertub is positive
    if dataset=='cifar10':
        mean=torch.tensor([0.4914,0.4822,0.4465]).reshape([1,3,1,1]).cuda()
        std=torch.tensor([0.2470,0.2435,0.2616]).reshape([1,3,1,1]).cuda()
    elif dataset=='cifar100':
        mean=torch.tensor([0.5071,0.4866,0.4409]).reshape([1,3,1,1]).cuda()
        std=torch.tensor([0.2673,0.2564,0.2762]).reshape([1,3,1,1]).cuda()
    elif dataset=='svhn':
        mean=torch.tensor([0,0,0]).reshape([1,3,1,1]).cuda()
        std=torch.tensor([1,1,1]).reshape([1,3,1,1]).cuda()
    elif dataset=='FashionMNIST':
        mean=torch.tensor([0.286]).reshape([1,1,1,1]).cuda()
        std=torch.tensor([0.315]).reshape([1,1,1,1]).cuda()

        # mean=torch.tensor([0]).reshape([1,1,1,1]).cuda()
        # std=torch.tensor([1]).reshape([1,1,1,1]).cuda()
    raw_input=norm_input*std+mean
    lb=(raw_input-pertub).clamp(min=0).clamp(max=1)
    ub=(raw_input+pertub).clamp(max=1).clamp(min=0)
    lb=(lb-mean)/std
    ub=(ub-mean)/std
    if bound:
        return lb,ub
    else:
        return ub

def compute_slim_loss(model):
    gamma_cnt=0
    slim_loss=0
    name_dict=model.named_modules()

    for name,m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            gamma_cnt+=m.weight.nelement()
            slim_loss+=m.weight.abs().sum()
        elif isinstance(m, BoundBatchNormalization):
            weight=m.inputs[1]
            gamma_cnt+=weight.param.nelement()
            slim_loss+=weight.param.abs().sum()
    slim_loss/=(1e-8+gamma_cnt)
    return slim_loss

def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        # normalization = NormalizeByChannelMeanStd(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'cifar100':
        classes = 100
        # normalization = NormalizeByChannelMeanStd(
        #   mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'svhn':
        classes = 10
        train_set_loader, test_loader = svhn_dataloaders(batch_size=args.batch_size, data_dir=args.data,
                                                             num_workers=args.workers)

    elif args.dataset == 'FashionMNIST':
        classes = 10
        train_set_loader, test_loader = FashionMNIST_dataloaders(batch_size=args.batch_size, data_dir=args.data,
                                                         num_workers=args.workers)


    else:
        raise ValueError('Dataset not supprot yet !')

    model = model_dict[args.arch](num_classes=classes, bn=args.bn, greyscale=('MNIST' in args.dataset))

    # model.normalize = normalization
    logger.info(model)

    return model, train_set_loader, test_loader



def load_model(model_ori,load_path,args,compress=False):
    """
    Load the model architectures and weights
    """
    # You can customize this function to load your own model based on model name.

    model_state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    if 'state_dict' in model_state_dict:
        model_state_dict = model_state_dict['state_dict']
    if type(model_state_dict) == list:
        model_state_dict = model_state_dict[0]
    elif type(model_state_dict) == OrderedDict or type(model_state_dict) == dict:
        orig_keys = []
        for x in model_state_dict.keys():
            if 'orig' in x:
                orig_keys.append(x)
        weights = [x.split('_')[0] for x in orig_keys]
        for x in weights:
            model_state_dict[x] = model_state_dict[x + '_orig'] * model_state_dict[x + '_mask']
            del model_state_dict[x + '_orig']
            del model_state_dict[x + '_mask']
    else:
        raise NotImplementedError

    model_ori.load_state_dict(model_state_dict)

    model_ori.eval()
    if not compress:
        return model_ori
    else:
        channel_nums=[]
        remain_channel_idxs=[]
        pruned_channel_idxs=[]
        conv_idxs=[]
        for i,layer in enumerate(model_ori):
            if isinstance(layer,nn.Conv2d):
                conv_idxs.append(i)
            if isinstance(layer,nn.BatchNorm2d):
                remain_channel_idxs.append(torch.where(layer.weight!=0)[0])
                pruned_channel_idxs.append(torch.where(layer.weight==0)[0])
                channel_nums.append(len(remain_channel_idxs[-1]))


        if args.dataset in ['cifar10','svhn','FashionMNIST','MNIST']:
            classes=10
        elif args.dataset=='cifar100':
            classes=100
        try:
            assert(sum([v==0 for v in channel_nums])==0)
        except AssertionError:
            print('empty layer detected, channel numbers:',channel_nums)
            raise AssertionError
        model_new = model_dict[args.arch](num_classes=classes, bn=args.bn, channels=channel_nums,greyscale=('MNIST' in args.dataset))
        for i in range(len(conv_idxs)):
            idx=conv_idxs[i]
            conv_=model_ori[idx]
            bn_=model_ori[idx+1]
            conv=model_new[idx]
            bn=model_new[idx+1]
            if i==0:
                prev_remain_idx=[0,1,2]
            else:
                prev_remain_idx=remain_channel_idxs[i-1]
            conv.weight.data.copy_(conv_.weight.data[remain_channel_idxs[i]][:,prev_remain_idx])
            conv.bias.data.copy_(conv_.bias.data[remain_channel_idxs[i]])
            bn.running_mean.data.copy_(bn_.running_mean.data[remain_channel_idxs[i]])
            bn.running_var.data.copy_(bn_.running_var.data[remain_channel_idxs[i]])
            bn.weight.data.copy_(bn_.weight.data[remain_channel_idxs[i]])
            bn.bias.data.copy_(bn_.bias.data[remain_channel_idxs[i]])

        linear_idx=conv_idxs[-1]+4
        linear_=model_ori[linear_idx]
        linear=model_new[linear_idx]
        #copy weight
        channel_=model_ori[conv_idxs[-1]].out_channels
        channel=channel_nums[-1]
        out_channel=linear_.out_features
        resolution=int(math.sqrt(linear_.in_features/channel_))
        weight_=linear_.weight.data.view(out_channel,channel_,resolution,resolution)
        weight=linear.weight.data.view(out_channel,channel,resolution,resolution)
        weight.copy_(weight_[:,remain_channel_idxs[-1]])
        #copy bias
        linear.bias.data.copy_(linear_.bias.data)

        #copy last linear layer
        linear1_idx=conv_idxs[-1]+6
        model_new[linear1_idx].load_state_dict(model_ori[linear1_idx].state_dict())

        model_new.eval()
        return model_new


