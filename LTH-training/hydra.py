import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

from main_imp import train
from lirpa_trainer import lirpa_train
from auto_LiRPA.eps_scheduler import FixedScheduler
from auto_LiRPA.bound_ops import BoundParams,BoundConv
from torch import autograd
from auto_LiRPA import BoundDataParallel
from pruner import prune_model_custom,remove_prune
from modules import Flatten


#code based on https://github.com/inspire-group/hydra

def prune_model_hydra(args,model,remain_ratio,train_loader,criterion=None):
    print("Start hydra pruning")
    epochs=args.hydra.epochs
    model_hydra,freezed_keys=convert_to_hydra(model)
    model_hydra.cuda()

    set_prune_rate_model(model_hydra,remain_ratio)
    if args.hydra.optimizer=='adam':
        optimizer=torch.optim.Adam(model_hydra.parameters(), args.hydra.lr)
    elif args.hydra.optimizer=='sgd':
        optimizer=torch.optim.SGD(model_hydra.parameters(), args.hydra.lr,momentum=args.hydra.momentum)
    optimizer.zero_grad()
    decreasing_lr = list(map(int, args.hydra.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    for epoch in range(epochs):
        print("=======>hydra training epoch:",epoch)
        if isinstance(model, nn.Sequential):
            train(args,train_loader, model_hydra, criterion, optimizer, epoch)
        elif isinstance(model,BoundDataParallel):
            lirpa_train(args,model_hydra,epoch,train_loader,FixedScheduler(args.eps),args.norm,True,optimizer,'CROWN-IBP',loss_fusion=not args.no_loss_fusion)
        scheduler.step()

    for m in model_hydra.modules():
        if hasattr(m,'subnet_mask_out'):
            if isinstance(m, BoundParams) and m.hydra == False:
                continue
            m.subnet_mask_out()

    optimizer.zero_grad()
    load_state_dict_from_hydra(model,model_hydra,freezed_keys)

def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            if isinstance(v, BoundParams) and v.hydra==False:
                continue
            v.set_prune_rate(prune_rate)

def load_state_dict_from_hydra(model_ori,model_hydra,freezed_keys=None):
    # if isinstance(model_ori,nn.Sequential):
    #     state_dict=model_hydra.state_dict()
    # elif isinstance(model_ori,BoundDataParallel):
    #     state_dict=model_hydra.module.state_dict()
    # for key in list(state_dict.keys()):
    #     if 'popup_scores' in key:
    #         del state_dict[key]
    # if isinstance(model_ori,nn.Sequential):
    #     model_ori.load_state_dict(state_dict)
    # elif isinstance(model_ori,BoundDataParallel):
    #     model_ori.module.load_state_dict(state_dict)

    mask_dict={}
    for n,m in model_hydra.named_modules():
        if isinstance(m,SubnetConv):
            mask_dict[n+'.weight_mask']=GetSubnet.apply(m.popup_scores.abs(), m.k)
        if isinstance(m,BoundParams) and m.hydra==True:
            mask_dict[m.ori_name+'_mask']=GetSubnet.apply(m.popup_scores.abs(),m.k)

    if isinstance(model_hydra,BoundDataParallel):
        for n,m in model_hydra.named_modules():
            if isinstance(m,BoundParams) and m.hydra==True:
                m.set_hydra_mode(False)
        for n, p in model_hydra.named_parameters():
            if n in freezed_keys:
                p.requires_grad = True

    prune_model_custom(model_ori,mask_dict)

def load_state_dict_to_hydra(model_ori,model_hydra):
    if isinstance(model_ori,nn.Sequential):
        state_dict=model_ori.state_dict()

        for n, m in model_ori.named_modules():
            if isinstance(m, nn.Conv2d):
                state_dict[n + '.popup_scores'] = torch.ones_like(m.weight)

        model_hydra.load_state_dict(state_dict)
    #
    # elif isinstance(model_ori,BoundDataParallel):
    #     state_dict=model_ori.module.state_dict()
    #     hydra_state_dict=model_hydra.module.state_dict()
    #
    #     for key in hydra_state_dict.keys():
    #         if 'popup_scores' in key:
    #             state_dict[key]=torch.ones_like(hydra_state_dict[key])
    #
    #     model_hydra.module.load_state_dict(state_dict)

def initialize_scaled_score(model_hydra):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model_hydra.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming uniform init
            if isinstance(m,SubnetConv):
                weight=m.weight
            elif isinstance(m,BoundParams):
                weight=m.param
            m.popup_scores.data = (
                math.sqrt(6 / n) * weight.data / torch.max(torch.abs(weight.data))
            )


def convert_to_hydra(model_ori):
    #model_ori need to be nn.Sequential or BoundDataParallel instance
    remove_prune(model_ori)

    if isinstance(model_ori, nn.Sequential):
        layer_list=[]
        for i,layer in enumerate(model_ori):
            if isinstance(layer,nn.Conv2d):
                layer_list.append(SubnetConv(layer.in_channels,
                                             layer.out_channels,
                                             layer.kernel_size,
                                             layer.stride,
                                             layer.padding,
                                             layer.dilation,
                                             layer.groups,
                                             layer.bias is not None))
            elif isinstance(layer,nn.Linear):
                layer_list.append(nn.Linear(layer.in_features,
                                               layer.out_features,
                                               layer.bias is not None))
            elif isinstance(layer,Flatten) or isinstance(layer,nn.ReLU):
                layer_list.append(type(layer)())
            elif isinstance(layer,nn.BatchNorm2d):
                layer_list.append(type(layer)(layer.num_features))
            else:
                raise NotImplementedError('hydra layer for '+str(type(layer))+ ' is not implemented.')

        model_hydra=nn.Sequential(*layer_list)
        load_state_dict_to_hydra(model_ori, model_hydra)

    elif isinstance(model_ori,BoundDataParallel):
        model_hydra=model_ori
        for name, m in model_hydra.named_modules():
            if isinstance(m, BoundConv):
                m.inputs[1].set_hydra_mode(True)

    initialize_scaled_score(model_hydra)

    #freeze parameters
    freezed_keys=[]
    for n,p in model_hydra.named_parameters():
        if 'popup_scores' not in n:
            p.requires_grad=False
            freezed_keys.append(n)

    return model_hydra,freezed_keys



class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def subnet_mask_out(self):
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.weight.data=adj*self.weight
