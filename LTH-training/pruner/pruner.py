from collections import defaultdict
import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from auto_LiRPA.bound_ops import *
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from torch.nn.parameter import Parameter
from utils import *
from grad_align import get_fgsm_grad_align_loss
from lirpa_trainer import lirpa_train

__all__  = ['prune_model_unstructured', 'prune_model_score_based','prune_model_structured', 'prune_model_imp_refill', 'prune_model_structured', 'prune_model_custom', 'remove_prune',
            'extract_mask', 'reverse_mask', 'check_sparsity', 'check_sparsity_dict']
#
# class DummyLayer(nn.Module):
#     def __init__(self,tensor):
#         super().__init__()
#         self.weight=nn.Parameter(tensor)

# def bound_apply_mask(module,input):
#     if hasattr(module,'param_mask'):
#         module.param=module.param_orig*module.param_mask
#
# def prune_multi_tensor(tensor_dict,px):
#     #globally pruning multiple tensors
#     # return a mask_dict with the corresponding names as in tensor_dict
#     keys=[]
#     tensors=[]
#     for x in tensor_dict:
#         keys.append(x)
#         tensors.append(tensor_dict[x])
#     layers=[]
#     for x in tensors:
#         layers.append(DummyLayer(x))
#     parameters=list(zip(layers,['weight']*len(layers)))
#     prune.global_unstructured(parameters,prune.L1Unstructured,amount=px)
#     mask_dict={}
#     for i,x in enumerate(layers):
#         mask_dict[keys[i]]=layers[i].weight_mask
#     return mask_dict
#
# def prune_unstable_weights(model,px):
#     #prune most unstable weights based on RS loss
#     #1-rs_loss
#     tensor_dict={}

def generate_score_based_mask(score_dict,remain_ratio):
    #pruning parameters with less scores
    global_scores = torch.cat([torch.flatten(v) for v in score_dict.values()])
    k = int((1.0 - remain_ratio) * global_scores.numel())
    mask_dict={}
    threshold, _ = torch.kthvalue(global_scores, k)
    for key in score_dict:
        mask_dict[key]=torch.where(score_dict[key]<=threshold,0,1)
    return mask_dict

def prune_model_score_based(args, model,remain_ratio,train_mode,prune_mode='snip',one_shot=False,**kwargs):
    #initialize mask
    prune_model_unstructured(model,0)
    name_dict=dict(model.named_modules())
    model.train()

    if prune_mode=='snip':

        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                m.weight_mask.requires_grad=True
            elif isinstance(m, BoundParams) and isinstance(name_dict['module.' + m.output_name[0]], BoundConv) and ('weight' in m.ori_name):
                m.param_mask.requires_grad=True

    # compute gradient
    if train_mode=='fgsm':
        image, target = next(iter(kwargs['dataloader']))
        image = image.cuda()
        target = target.cuda()
        loss,_=get_fgsm_grad_align_loss(args,kwargs['criterion'],model,image,target,kwargs['optimizer'],kwargs['eps'])
    elif train_mode=='cibp':
        loss = lirpa_train(args, model, 1, kwargs['dataloader'], kwargs['eps_scheduler'], kwargs['norm'], True, kwargs['optimizer'], kwargs['bound_type'], loss_fusion=kwargs['loss_fusion'],prune=True)

    if prune_mode=='grasp':
        temp=200
        masked_parameters=[]
        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                masked_parameters.append(m.weight_orig)
            elif isinstance(m,BoundConv):
                masked_parameters.append(m.inputs[1].param_orig)
        grads=torch.autograd.grad(loss, masked_parameters, create_graph=False)
        stopped_grads=torch.cat([g.reshape(-1) for g in grads if g is not None])
        if train_mode == 'fgsm':
            loss, _ = get_fgsm_grad_align_loss(args, kwargs['criterion'], model, image, target,
                                               kwargs['optimizer'], kwargs['eps'])
        elif train_mode == 'cibp':
            loss = lirpa_train(args, model, 1, kwargs['dataloader'], kwargs['eps_scheduler'], kwargs['norm'], True,
                                   kwargs['optimizer'], kwargs['bound_type'], loss_fusion=kwargs['loss_fusion'],
                                   prune=True)

    loss.backward()

    score_dict={}
    #generate scores
    if prune_mode=='snip':
        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                score_dict[name+'.weight_mask']=torch.clone(m.weight_mask.grad).detach().abs_()
                m.weight_orig.grad.data.zero_()
                m.weight_mask.grad.data.zero_()
                m.weight_mask.requires_grad=False
            elif isinstance(m, BoundParams) and isinstance(name_dict['module.' + m.output_name[0]], BoundConv) and ('weight' in m.ori_name):
                score_dict[m.ori_name+'_mask']=torch.clone(m.param_mask.grad).detach().abs_()
                m.param_orig.grad.data.zero_()
                m.param_mask.grad.data.zero_()
                m.param_mask.requires_grad = False
    elif prune_mode=='taylor1ScorerAbs':
        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                score_dict[name+'.weight_mask']=torch.clone(m.weight_orig.grad*m.weight_orig).detach().abs_()
                m.weight_orig.grad.data.zero_()
            elif isinstance(m, BoundParams) and isinstance(name_dict['module.' + m.output_name[0]], BoundConv) and ('weight' in m.ori_name):
                score_dict[m.ori_name+'_mask']=torch.clone(m.param_orig.grad*m.param_orig).detach().abs_()
                m.param_orig.grad.data.zero_()


    pre_mask_dict=extract_mask(model)
    for key in score_dict:
        score_dict[key]=pre_mask_dict[key]*score_dict[key]

    mask_dict=generate_score_based_mask(score_dict,remain_ratio)
    remove_prune(model)
    prune_model_custom(model,mask_dict)
    model.eval()

    if one_shot:
        remove_prune(model)

def prune_model_unstructured(model, rate, method='random'):

    logger.info('Apply Unstructured L1 Pruning Globally')
    name_dict=dict(model.named_modules())

    names=[]
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))
        elif isinstance(m,BoundParams) and isinstance(name_dict['module.'+m.output_name[0]],BoundConv) and ('weight' in m.ori_name):
            #proxy conv layer for pruning
            conv=nn.Conv2d(m.param.shape[1],m.param.shape[0],m.param.shape[2:])
            if hasattr(m,'param_mask'):
                conv.weight.data=m.param_orig.detach()
                prune.CustomFromMask.apply(conv,'weight',m.param_mask.detach())
            else:
                conv.weight.data=m.param.detach()
            parameters_to_prune.append((conv,'weight'))
            names.append(name)

    pruning_method=None
    if method=='random':
        pruning_method=prune.RandomUnstructured
    elif method=='l1unstruct':
        pruning_method=prune.L1Unstructured

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=rate,
    )

    if isinstance(model,BoundDataParallel):
        for i,name in enumerate(names):
            param_orig=Parameter(parameters_to_prune[i][0].weight_orig.detach(),requires_grad=True)
            name_dict[name].register_parameter('param_orig',param_orig)
            name_dict[name].register_buffer('param_mask',parameters_to_prune[i][0].weight_mask.detach())
            del name_dict[name].param
            name_dict[name].param=name_dict[name].param_orig*name_dict[name].param_mask
            # name_dict[name].register_forward_pre_hook(bound_apply_mask)



def prune_model_imp_refill(model, imp_rate, channel_remain_ratio, local=False):

    logger.info('Apply IMP-Refill Pruning')
    prune_model_unstructured(model,imp_rate,method='l1unstruct')
    prune_model_structured(model,channel_remain_ratio,mode='refill',local=local)


def prune_model_ln_structured(model,rate):
    logger.info('Apply Structured L1 Pruning ')

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m,'weight',amount=rate,n=1,dim=1)


def prune_model_structured(model,remain_ratio,mode='slim',local=False,loss=None):
    logger.info('Apply Structured Pruning')
    #model must be nn.Sequential FeedFroward Network, where each nn.Conv2d followed with nn.BatchNorm2D

    scores={}
    for i,(name,m) in enumerate(model.named_modules()):
        if isinstance(m,nn.BatchNorm2d):
            if mode=='slim':
                scores[name]=m.weight.detach().abs_()
            elif mode=='nrs':
                print('max_loss',loss[name].max(),'min_loss',loss[name].min())
                # scores[name]=(m.weight!=0).float()*(1-loss[name]+torch.rand_like(loss[name])*1e-3)#convert nrs loss to scores
                scores[name]=(m.weight!=0).float()*(-loss[name])#convert nrs loss to scores
            elif mode=='refill':
                conv=model[i-2]
                scores[name]=conv.weight.detach().abs_().mean(axis=[1,2,3])
        elif isinstance(m,BoundBatchNormalization):
            if mode=='slim':
                scores[id(m)]=m.inputs[1].param.detach().abs_()
            elif mode=='refill':
                conv_weight=m.inputs[0].inputs[1].param
                scores[id(m)]=conv_weight.detach().abs_().mean(axis=[1,2,3])
        elif isinstance(m,BoundRelu) and isinstance(m.inputs[0],BoundBatchNormalization):
            if mode=='nrs':
                bn=m.inputs[0]
                # scores[id(bn)]=(1-loss[id(bn)]+torch.rand_like(loss[id(bn)])*1e-3)*(bn.inputs[1].param!=0)#convert nrs loss to scores
                scores[id(bn)]=(-loss[id(bn)])*(bn.inputs[1].param!=0)#convert nrs loss to scores



    k=1-remain_ratio
    global_scores=torch.cat([torch.flatten(v) for v in scores.values()])
    global_thres,_=torch.kthvalue(global_scores,k=int(k*global_scores.nelement()))
    local_thres = {}
    for key in scores:
        thres,_=torch.kthvalue(scores[key],k=int(k*scores[key].nelement()))
        local_thres[key]=thres

    mask_dict={}
    for i,(name,m) in enumerate(model.named_modules()):
        if isinstance(m,nn.BatchNorm2d):
            mask=torch.where(scores[name]<=(local_thres[name] if local else global_thres),0,1).detach()
            if mode=='refill':
                conv=model[i-2]
                conv.weight_mask=mask.detach().view(-1,1,1,1).expand_as(conv.weight_mask)
            mask_dict[name+'.weight_mask']=mask
            mask_dict[name+'.bias_mask']=mask
        elif isinstance(m,BoundBatchNormalization):
            weight=m.inputs[1]
            bias=m.inputs[2]
            mask=torch.where(scores[id(m)]<=(local_thres[id(m)] if local else global_thres),0,1).detach()
            if mode=='refill':
                conv_weight=m.inputs[0].inputs[1]
                conv_weight.param_mask=mask.detach().view(-1,1,1,1).expand_as(conv_weight.param_mask)
            mask_dict[weight.ori_name+'_mask']=mask
            mask_dict[bias.ori_name+'_mask']=mask

    if mode!='refill':
        remove_prune(model)
    prune_model_custom(model,mask_dict)


def prune_model_custom(model, mask_dict):

    logger.info('Pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Linear):
            mask_name = name+'.weight_mask'
            if mask_name in mask_dict:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            mask_name = name + '.bias_mask'
            if mask_name in mask_dict:
                prune.CustomFromMask.apply(m, 'bias', mask=mask_dict[name + '.bias_mask'])
        if isinstance(m,BoundParams):
            mask_name = m.ori_name+'_mask'
            if mask_name in mask_dict:
                param_orig = Parameter(m.param.detach().cuda(), requires_grad=True)
                m.register_parameter('param_orig', param_orig)
                m.register_buffer('param_mask', mask_dict[mask_name].cuda())
                del m.param
                m.param = m.param_orig * m.param_mask
                # m.register_forward_pre_hook(bound_apply_mask)
                # BoundParams uses forward instead of __call__, thus no effect for pre_hook



def remove_prune(model):

    logger.info('Remove hooks for multiplying masks')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.Linear):
            if hasattr(m,'weight_mask'):
                prune.remove(m,'weight')
            if hasattr(m,'bias_mask'):
                prune.remove(m,'bias')
        if isinstance(m,BoundParams):
            if hasattr(m,'param_mask'):
                del m.param
                m.register_parameter('param',Parameter((m.param_orig*m.param_mask).detach(),requires_grad=True))
                del m.param_mask
                del m.param_orig
                # m._forward_pre_hooks.clear()



# Mask operation function
def extract_mask(model):

    #for normal pruning
    model_dict = model.state_dict()
    new_dict = {}
    has_mask=False
    for key in model_dict.keys():
        if 'mask' in key:
            has_mask=True
            new_dict[key] = copy.deepcopy(model_dict[key])

    #for hydra pruning
    if not has_mask:
        for n,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                new_dict[n+'.weight_mask']=(m.weight!=0).int()

    return new_dict

def reverse_mask(mask_dict):

    new_dict = {}
    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

# Mask statistic function
def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0
    layerwise_sparsities=[]
    pruned_channels=0
    total_channels=0

    if isinstance(model,BoundDataParallel):
        name_dict=dict(model.named_modules())

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))
            layerwise_sparsities.append(1-float(torch.sum(m.weight==0)/m.weight.nelement()))
        elif isinstance(m,BoundParams) and isinstance(name_dict['module.'+m.output_name[0]],BoundConv) and ('weight' in m.ori_name):
            sum_list = sum_list + float(m.param.nelement())
            zero_sum = zero_sum + float(torch.sum(m.param == 0))
            layerwise_sparsities.append(1 - float(torch.sum(m.param == 0) / m.param.nelement()))
        elif isinstance(m,nn.BatchNorm2d):
            pruned_channels+=(m.weight==0).sum()
            total_channels+=m.weight.nelement()
        elif isinstance(m,BoundBatchNormalization):
            weight=m.inputs[1]
            pruned_channels+=(weight.param==0).sum()
            total_channels+=weight.param.nelement()


    remain_weight_ratie = 100*(1-zero_sum/sum_list)
    logger.info('* remain weight ratio = {}%'.format(remain_weight_ratie))

    channel_sparsity=100*(1-pruned_channels/total_channels)
    logger.info('* remain channel ratio = {}%'.format(channel_sparsity))

    return remain_weight_ratie,channel_sparsity,layerwise_sparsities

def check_sparsity_dict(state_dict):
    
    sum_list = 0
    zero_sum = 0

    for key in state_dict.keys():
        if 'mask' in key:
            sum_list += float(state_dict[key].nelement())
            zero_sum += float(torch.sum(state_dict[key] == 0))  

    remain_weight_ratie = 100*(1-zero_sum/sum_list)
    if zero_sum:
        logger.info('* remain weight ratio = {}%'.format(100*(1-zero_sum/sum_list)))
    else:
        logger.info('no weight for calculating sparsity')

    return remain_weight_ratie

