import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from auto_LiRPA.bound_ops import BoundRelu,BoundBatchNormalization,BoundConv,BoundLinear,BoundAdd,Bound
from auto_LiRPA import BoundDataParallel
from modules import Flatten
import math


def interval_bound_prop(layer,lb_,ub_):
    lb,ub=None,None
    if isinstance(layer,nn.Linear):
        mu_=(ub_+lb_)/2
        r_=(ub_-lb_)/2
        mu=layer(mu_)
        r=F.linear(r_,torch.abs(layer.weight),None)
        lb=mu-r
        ub=mu+r
    elif isinstance(layer,nn.Conv2d):
        mu_ = (ub_ + lb_) / 2
        r_ = (ub_ - lb_) / 2
        mu = layer(mu_)
        r = F.conv2d(r_,torch.abs(layer.weight),None,layer.stride,layer.padding,layer.dilation,layer.groups)
        lb = mu - r
        ub = mu + r
    elif isinstance(layer,nn.BatchNorm2d):
        mid_=(lb_+ub_)/2
        diff_=(ub_-lb_)/2
        mean=layer.running_mean
        var=layer.running_var
        weight=layer.weight
        bias=layer.bias
        tmp_weight=weight/torch.sqrt(var+1e-8)
        tmp_bias=bias-mean*tmp_weight
        shape=(1,-1,1,1)
        center=tmp_weight.view(*shape)*mid_+tmp_bias.view(*shape)
        deviation=torch.abs(tmp_weight).view(*shape)*diff_
        lb=center-deviation
        ub=center+deviation
    elif isinstance(layer,nn.ReLU):
        lb=F.relu(lb_)
        ub=F.relu(ub_)
    elif isinstance(layer,Flatten):
        lb=layer(lb_)
        ub=layer(ub_)
    else:
        raise NotImplementedError(type(layer),' IBP is not implemented!')
    return lb,ub

def compute_bound(model,lb_,ub_,gamma_pow=2,stat_stability=False,return_nrs=False):
    #don't compute bound for unsupported model
    if not isinstance(model,nn.Sequential):
        ub_=lb_=model(lb_)
        loss=torch.tensor(0).cuda()
        unstable_ratio=0
        layerwise_losses=[0]
        layerwise_ratios=[0]
    else:
        layer_unstables = defaultdict(lambda: 0)
        layer_nelements = defaultdict(lambda: 0)
        layer_losses = defaultdict(lambda: 0)
        loss = 0
        nelement = 0
        layer_new_losses={}
        i_to_name={}
        i_to_module={}
        for i,(name,m) in enumerate(model.named_modules()):
            i_to_name[i]=name
            i_to_module[i]=m
            if isinstance(m,nn.Sequential):
                continue
            lb,ub=interval_bound_prop(m,lb_,ub_)
            if isinstance(m,nn.ReLU):
                nelement+=lb_.nelement()
                if isinstance(i_to_module[i-1],nn.BatchNorm2d):
                    gamma=i_to_module[i-1].weight.view(1,-1,1,1)
                    loss+=-torch.tanh(1+lb_*ub_/(torch.pow(gamma.detach(),gamma_pow)+1e-20)).sum()
                    if return_nrs:
                        layer_new_losses[i_to_name[i-1]]=(-torch.tanh(1+lb_*ub_/(torch.pow(gamma.detach(),gamma_pow)+1e-20))).mean(axis=[0,2,3]).detach()
                else:
                    loss+=-torch.tanh(1+lb_*ub_).sum()
                if stat_stability:
                    lb_approx=(torch.abs(lb_)>1e-5)*lb_
                    ub_approx=(torch.abs(ub_)>1e-5)*ub_
                    layer_unstables[i-1]+=int(((lb_approx*ub_approx)<0).sum())
                    layer_nelements[i-1]+=int(lb_.nelement())
                    layer_losses[i-1]+=-torch.tanh(1+lb_*ub_).sum()
            lb_=lb
            ub_=ub

        loss=loss/(1e-8+nelement)
        layerwise_ratios=[]
        unstable_ratio=0
        unstable_cnt=0
        layerwise_losses=[]
        for x in layer_unstables.keys():
            layerwise_ratios.append(float(layer_unstables[x]/layer_nelements[x]))
            unstable_cnt+=layer_unstables[x]
            layerwise_losses.append(float(layer_losses[x]/layer_nelements[x]))
        unstable_ratio=unstable_cnt/(1e-8+nelement)

    if not return_nrs:
        return lb_,ub_,loss,unstable_ratio,layerwise_losses,layerwise_ratios
    else:
        return lb_,ub_,loss,unstable_ratio,layerwise_losses,layerwise_ratios,layer_new_losses

def compute_lirpa_nrs(model,gamma_pow=2):
    #compute nrs for auto_lirpa model
    #return: nrs_loss: scalar
    #        nrs_dict: {id(BoundRelu): channel nrs loss}
    nrs_dict={}
    instab_dict={}
    for name,m in model.named_modules():
        if isinstance(m,BoundRelu) and isinstance(m.inputs[0],BoundBatchNormalization):
            bn=m.inputs[0]
            gamma=bn.inputs[1].param.view(1,-1,1,1)
            nrs_dict[id(bn)] = -torch.tanh(1 + bn.lower * bn.upper / (torch.pow(gamma.detach(), gamma_pow) + 1e-8)).mean(axis=[0, 2, 3])
            # nrs_dict[id(bn)]=-torch.tanh(1+stability).mean(axis=[0,2,3])

            #stability=(bn.lower*bn.upper/(torch.pow(gamma,gamma_pow)+1e-20)).detach()
            # instab_dict[id(bn)]=(-stability).mean(axis=[0,2,3]).detach()
            instab_dict[id(bn)]=torch.ones_like(gamma)


    nrs_loss=torch.cat([v for v in nrs_dict.values()]).mean()
    for key in nrs_dict:
        nrs_dict[key]=nrs_dict[key].detach()
    return nrs_loss,nrs_dict,instab_dict

def stat_instability(model):
    # compute relu instability for auto_lirpa model
    # return: unstable_ratio: scalar
    #        nrs_dict: {id(BoundRelu): scale unstable ratio}
    layer_unstable_dict={}
    neuron_unstable_list=[]
    neuron_med_list=[]
    neuron_med_sqr_list=[]
    pre_bn_interval_list=[]
    pre_act_interval_list=[]
    bn_neuron_unstable_list=[]
    bound_prod_list=[]
    bn_bound_prod_list=[]
    for name,m in model.named_modules():
        if isinstance(m,BoundRelu) and isinstance(m.inputs[0],BoundBatchNormalization):
            lb=m.inputs[0].lower
            ub=m.inputs[0].upper
            lb_approx = (torch.abs(lb) > 1e-5) * lb
            ub_approx = (torch.abs(ub) > 1e-5) * ub
            layer_unstable_dict[id(m)] = ((lb_approx * ub_approx) < 0).float().mean(axis=[0])
            neuron_unstable_list.append(layer_unstable_dict[id(m)])

            med=(lb+ub)/2
            neuron_med=med.mean(axis=0)
            neuron_med_sqr=(med*med).mean(axis=0)
            neuron_med_list.append(neuron_med)
            neuron_med_sqr_list.append(neuron_med_sqr)

            pre_act_interval_list.append((ub-lb).mean(axis=0))
            bound_prod_list.append((-ub*lb).mean(axis=0))

            lb_=m.inputs[0].inputs[0].lower
            ub_=m.inputs[0].inputs[0].upper
            bn_bound_prod_list.append((-ub_*lb_).mean(axis=0))
            pre_bn_interval_list.append((ub_-lb_).mean(axis=0))
            lb_approx_ = (torch.abs(lb_) > 1e-5) * lb_
            ub_approx_ = (torch.abs(ub_) > 1e-5) * ub_
            bn_neuron_unstable_list.append(((lb_approx_ * ub_approx_) < 0).float().mean(axis=[0]))

    unstable_ratio=torch.cat([v.mean(axis=[1,2]) for v in layer_unstable_dict.values()]).mean().detach()
    for key in layer_unstable_dict:
         layer_unstable_dict[key]=float(layer_unstable_dict[key].mean())

    neuron_unstable_ratio=torch.cat([v.view(-1) for v in neuron_unstable_list]).detach()
    bn_neuron_unstable_ratio=torch.cat([v.view(-1) for v in bn_neuron_unstable_list]).detach()
    neuron_med=torch.cat([v.view(-1) for v in neuron_med_list]).detach()
    neuron_med_sqr=torch.cat([v.view(-1) for v in neuron_med_sqr_list]).detach()
    pre_act_interval=torch.cat([v.view(-1) for v in pre_act_interval_list]).detach()
    pre_bn_interval=torch.cat([v.view(-1) for v in pre_bn_interval_list]).detach()
    bound_prod=torch.cat([v.view(-1) for v in bound_prod_list]).detach()
    bn_bound_prod=torch.cat([v.view(-1) for v in bn_bound_prod_list]).detach()



    return unstable_ratio,neuron_unstable_ratio,layer_unstable_dict,neuron_med,neuron_med_sqr,pre_act_interval,pre_bn_interval,bn_neuron_unstable_ratio,bound_prod,bn_bound_prod


#https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/main/regularization.py
def compute_fast_ibp_reg(model, eps_scheduler):
    loss = torch.zeros(()).cuda()
    tol=0.5
    # Handle the non-feedforward case
    l0 = torch.zeros_like(loss)
    loss_tightness, loss_std, loss_relu, loss_ratio = (l0.clone() for i in range(4))


    if isinstance(model, BoundDataParallel):
        modules = list(model._modules.values())[0]._modules
    else:
        modules = model._modules
    node_inp = modules['/input.1']
    tightness_0 = ((node_inp.upper - node_inp.lower) / 2).mean()
    ratio_init = tightness_0 / ((node_inp.upper + node_inp.lower) / 2).std()
    cnt_layers = 0
    cnt = 0
    for m in model._modules.values():
        if isinstance(m, BoundRelu):
            lower, upper = m.inputs[0].lower, m.inputs[0].upper
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()
            std_ = center.std()

            loss_tightness += F.relu(tol - tightness_0 / tightness.clamp(min=1e-12)) / tol
            loss_std += F.relu(tol - std_) / tol
            cnt += 1

            # L_{relu}
            mask_act, mask_inact = lower>0, upper<0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()# / center.numel()
            var_inact = (delta * mask_inact).sum()# / center.numel()

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = ((
                F.relu(tol - mean_ratio) + F.relu(tol - var_ratio))
                / tol)
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_

    loss_tightness /= cnt
    loss_std /= cnt
    loss_relu /= cnt

    loss += loss_tightness+loss_relu

    reg_lambda=0.5
    intensity = reg_lambda * (1 - eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
    loss *= intensity

    return loss


def get_params(model):
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_init(model_ori, model):
    weights, biases = get_params(model_ori)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')
    for node in model._modules.values():
        if isinstance(node, BoundConv) or isinstance(node, BoundLinear):
            if len(node.inputs[0].inputs) > 0 and isinstance(node.inputs[0].inputs[0], BoundAdd):
                print(f'Adjust weights for node {node.name} due to residual connection')
                node.inputs[1].param.data /= 2