'''
    main process for a Lottery Tickets experiments
'''
import os
import pdb
import readline
import time
import pickle
import random
import shutil
import argparse
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore',ResourceWarning)
import logging
from setproctitle import setproctitle

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd
#import apex.amp as amp

cpu_num = 2 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

from utils import *
from pruner import *
from bound_prop import *
import hydra
import grad_align
from grad_align import rob_acc, l2_norm_batch, get_input_grad, clamp, get_fgsm_grad_align_loss
from auto_LiRPA.eps_scheduler import AdaptiveScheduler, FixedScheduler, SmoothedScheduler


parser = argparse.ArgumentParser(description='FSGM Experiments')
parser.add_argument('--config', default='configs/fgsm_random.yml', type=str, help='must specify a config file')

##################################### Training setting #################################################
parser.add_argument('--optimizer', type=str, default='sgd', help='optimier: adam / sgd')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

##################################### adv training setting #################################################
parser.add_argument('--grad_align_cos_lambda', default=0.2, type=float,help='coefficient of the cosine gradient alignment regularizer')
parser.add_argument('--fgsm_alpha', default=1.25, type=float)

##################################### Pruning setting #################################################

args=set_env(parser)
setproctitle(args.save_dir)

best_sa = 0

def main():
    global args, best_sa
    print(logger.handlers)
    logger.info(args)

    # torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    if args.prune_type == 'lt':
        logger.info('lottery tickets setting (rewind to the same random init)')
        initialization = deepcopy(model.state_dict())
    elif args.prune_type == 'pt':
        logger.info('lottery tickets from best dense weight')
        initialization = None
    elif args.prune_type == 'rewind_lt':
        logger.info('lottery tickets with early weight rewinding')
        initialization = None
    else:
        raise ValueError('unknown prune_type')

    if args.resume_state>0:
        load_path=os.path.join(args.save_dir,'%02d'%args.resume_state+'last.model')
        model=load_model(model,load_path,args)
        initialization=torch.load(load_path)['init_weight']

        current_mask = extract_mask(model)
        remove_prune(model)
        model.load_state_dict(initialization)
        prune_model_custom(model, current_mask)


    if args.optimizer=='adam':
        optimizer=torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)
    elif args.optimizer=='sgd':
        optimizer=torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_milestones, gamma=0.1)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)

    double_bp = True if args.grad_align_cos_lambda > 0 else False
    # if args.half_prec:
    #     if double_bp:
    #         amp.register_float_function(torch,'batch_norm')
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    #FIXME: make these variables resumable
    all_result = {}
    all_result['train_acc'] = []
    all_result['test_acc'] = []
    all_result['clean_acc'] = []
    all_result['unstable_ratio'] = []

    start_epoch = 0
    start_state = args.resume_state+1 if args.resume_state>0 else 0

    logger.info('######################################## Start Standard Training Iterative Pruning ########################################')
    acc_results={}
    acc_results['train']=[]
    acc_results['test'] = []
    acc_results['clean'] = []

    ratio_results=[]
    loss_results=[]
    layerwise_loss_results=[]
    layerwise_ratio_results=[]
    best_ratio=None
    best_loss=None
    best_layerwise_losses=None
    best_layerwise_ratios=None
    layerwise_sparsities=[]
    sparsities=[]
    channel_sparsities=[]
    layer_new_losses = None

    if args.one_shot:
        logger.info('score-based pruning')
        prune_model_score_based(args,model,  1-args.rate, 'fgsm',args.prune_method, True, criterion=criterion, dataloader=train_loader, optimizer=optimizer, eps=args.eps)

    for state in range(start_state, args.prune_times):

        logger.info('******************************************')
        logger.info('pruning state {}'.format(state))
        logger.info('******************************************')
        for x in acc_results:
            acc_results[x].append(0)

        sparsity,channel_sparsity,layerwise_sparsity=check_sparsity(model)
        sparsities.append(sparsity/100)
        channel_sparsities.append(channel_sparsity/100)
        for i,x in enumerate(layerwise_sparsity):
            if len(layerwise_sparsities)<=i:
                layerwise_sparsities.append([])
            layerwise_sparsities[i].append(x)
            # plot training curve
        plt.ylim(0,1)
        plt.plot(channel_sparsities, label='channels')
        plt.plot(sparsities, label='overall weights')
        for i in range(len(layerwise_sparsities)):
            plt.plot(layerwise_sparsities[i],label='layer '+str(i))
        plt.ylabel('REMAINING WEIGHT')
        plt.xlabel('prune_time@%.2f'%args.rate)
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'sparsity.png'))
        plt.close()

        for epoch in range(start_epoch, args.epochs):

            acc = train(args,train_loader, model, criterion, optimizer, epoch, eps_scheduler)

            if state == 0:
                if (epoch+1) == args.rewind_epoch:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                    if args.prune_type == 'rewind_lt':
                        initialization = deepcopy(model.state_dict())

            # evaluate on validation set
            # #clean test
            # val_clean_acc,v_rs_loss,v_unstable_ratio,_,_,layer_new_losses = validate(args,val_loader, model, criterion)
            # #fgsm attack
            # val_acc=validate(args,val_loader,model,criterion,attack='fgsm',optimizer=optimizer)

            # evaluate on test set
            #clean test
            test_clean_acc,t_rs_loss,t_unstable_ratio,layerwise_losses,layerwise_ratios,layer_new_losses = test(args,test_loader, model, criterion)
            # fgsm attack
            test_acc = test(args,test_loader, model, criterion, attack='fgsm',optimizer=optimizer)

            scheduler.step()

            all_result['train_acc'].append(acc)
            all_result['clean_acc'].append(test_clean_acc)
            all_result['test_acc'].append(test_acc)

            # remember best prec@1 and save checkpoint
            is_best_sa = test_acc  > best_sa
            best_sa = max(test_acc, best_sa)
            if is_best_sa:
                acc_results['train'][-1]=acc
                acc_results['test'][-1]=test_acc
                acc_results['clean'][-1]=test_clean_acc
                best_ratio=t_unstable_ratio
                best_loss=t_rs_loss
                best_layerwise_losses=layerwise_losses
                best_layerwise_ratios=layerwise_ratios

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initialization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)

            # plot training curve
            plt.ylim(10,100)
            plt.plot(all_result['train_acc'], label='train_acc')
            plt.plot(all_result['test_acc'], label='test_acc')
            plt.plot(all_result['clean_acc'], label='clean_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, '%02dnet_train.png'%state))
            plt.close()


        #report result
        check_sparsity(model)
        test_pick_best_epoch = np.argmax(np.array(all_result['test_acc']))
        logger.info('* best SA = {}, Epoch = {}'.format(all_result['test_acc'][test_pick_best_epoch], test_pick_best_epoch+1))

        if args.one_shot:
            break

        all_result = {}
        all_result['train_acc'] = []
        all_result['test_acc'] = []
        all_result['clean_acc'] = []
        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            logger.info('* loading pretrained weight')
            initialization = torch.load(os.path.join(args.save_dir, '0model_SA_best.pth.tar'), map_location = torch.device(args.device))['state_dict']

        if args.optimizer=='adam':
            optimizer=torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)
        elif args.optimizer=='sgd':
            optimizer=torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

        eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_milestones, gamma=0.1)
        if args.rewind_epoch:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()

        #pruning and rewind
        remain_ratio = math.pow(1 - args.rate, state + 1)
        if args.prune_method=='random':
            logger.info('random pruning')
            prune_model_unstructured(model, args.rate,'random')
        elif args.prune_method == 'l1unstruct':
            logger.info('L1 unstructrued pruning')
            prune_model_unstructured(model, args.rate,'l1unstruct')
        elif args.prune_method == 'snip' or args.prune_method=='taylor1ScorerAbs':
            logger.info('score-based pruning')
            prune_model_score_based(args, model, remain_ratio, 'fgsm', args.prune_method, False,
                                    criterion=criterion, dataloader=train_loader, optimizer=optimizer, eps=args.eps)
        elif args.prune_method=='l1-channel':
            logger.info('L1 channel pruning')
            prune_model_ln_structured(model,args.rate)
        elif args.prune_method=='refill':
            logger.info('IMP Refilling...')
            prune_model_imp_refill(model,args.imp_rate,remain_ratio,local=args.layerwise)
        elif args.prune_method == 'slim':
            logger.info('network slimming...')
            prune_model_structured(model, remain_ratio, mode='slim',local=args.layerwise)
        elif args.prune_method == 'nrs':
            logger.info('NRSLoss pruning')
            prune_model_structured(model,remain_ratio,mode='nrs',local=args.layerwise,loss=layer_new_losses)
        elif args.prune_method=='hydra':
            logger.info('Hydra robustness-based pruning')
            hydra.prune_model_hydra(args,model,remain_ratio,train_loader,criterion)
        else:
            raise NotImplementedError('Pruning method not implemented!')

        check_sparsity(model)

        # weight rewinding
        current_mask = extract_mask(model)
        remove_prune(model)
        model.load_state_dict(initialization)
        prune_model_custom(model, current_mask)


        plt.ylim(10,100)
        plt.plot(acc_results['train'], label='train')
        plt.plot(acc_results['test'], label='test')
        plt.plot(acc_results['clean'], label='test_clean')
        plt.xlabel('prune_time@%.2f'%args.rate)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'Accuracy.png'))
        plt.close()


        ratio_results.append(best_ratio)
        loss_results.append(best_loss)
        for i in range(len(best_layerwise_losses)):
            if len(layerwise_loss_results)<=i:
                layerwise_loss_results.append([])
            layerwise_loss_results[i].append(best_layerwise_losses[i])
            if len(layerwise_ratio_results)<=i:
                layerwise_ratio_results.append([])
            layerwise_ratio_results[i].append(best_layerwise_ratios[i])

        plt.ylim(0,1)
        plt.plot(ratio_results, label='overall ratio')
        for i,x in enumerate(layerwise_ratio_results):
            plt.plot(x,label='layer '+str(i))
        plt.xlabel('prune_time@%.2f'%args.rate)
        plt.ylabel('UNSTABLE RATIO')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'unstable_results.png'))
        plt.close()

        plt.ylim(-1, 1)
        plt.plot(loss_results, label='overall loss')
        for i,x in enumerate(layerwise_loss_results):
            plt.plot(x,label='layer '+str(i))
        plt.xlabel('prune_time@%.2f'%args.rate)
        plt.ylabel('RS LOSS')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'RS_loss.png'))
        plt.close()

    logger.info('Acc result:')
    logger.info(acc_results['clean'])
    logger.info('adv acc result:')
    logger.info(acc_results['test'])



def train(args,train_loader, model, criterion, optimizer, epoch, eps_scheduler=None):
    if eps_scheduler:
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(train_loader.dataset) + train_loader.batch_size - 1) / train_loader.batch_size))
        logger.info('eps: {}'.format(eps_scheduler.get_eps()))


    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        if eps_scheduler:
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
        else:
            eps = args.eps
        # if epoch < args.warmup:
        #     warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        loss,output = get_fgsm_grad_align_loss(args,criterion,model,image,target,optimizer,eps)

        if args.slim_weight>0:
            slim_loss=compute_slim_loss(model)
            loss+=args.slim_weight*slim_loss

        if args.prune_method=='nrs' or args.rs_reg:
            data_min,data_max=transform_pertub(args.dataset,image,args.eps,bound=True)
            lb,ub,rs_loss,unstable_ratio,layerwise_loss,layerwise_ratio=compute_bound(model,data_min,data_max,gamma_pow=args.gamma_pow)
            if args.rs_reg:
                loss += args.rs_weight*rs_loss

        optimizer.zero_grad()
        grad_align.backward(loss, optimizer, False)
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        # mt = torch.zeros(1).cuda()
        # for p in model.parameters():
        #     mt = torch.max(mt, p.grad.abs().max())
        # print('after', mt)
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    logger.info('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test(args, test_loader, model, criterion, attack=None,optimizer=None):
    """
    Run evaluation
    """
    double_bp = True if args.grad_align_cos_lambda > 0 else False

    losses = AverageMeter()
    top1 = AverageMeter()
    rs_losses = AverageMeter()
    unstable_ratios = AverageMeter()
    layerwise_losses= []
    layerwise_ratios= []

    if attack==None:
        layer_new_losses=defaultdict(lambda: 0)

    eps=args.eps
    # switch to evaluate mode
    model.eval()
    for i, (image, target) in enumerate(test_loader):
        
        image = image.cuda()
        target = target.cuda()

        if attack=='fgsm':
            delta = torch.zeros_like(image, requires_grad=True)

            # compute output
            tmp_delta=transform_pertub(args.dataset,image,delta)
            output_clean = model(image+tmp_delta)
            loss = criterion(output_clean, target)

            # half-precision in fgsm for training speedup
            # if args.half_prec:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if double_bp else False)[0]
            #         grad /= scaled_loss / loss  # reverse back the scaling
            # else:
            grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]

            grad = grad.detach()
            argmax_delta = eps * grad_align.sign(grad)
            delta.data = grad_align.clamp(delta.data + args.fgsm_alpha * argmax_delta,-eps,eps)
            tmp_delta = transform_pertub(args.dataset,image, delta)
            delta.data = tmp_delta.data - image
            delta = delta.detach()

        # compute output
        with torch.no_grad():
            if attack==None:
                output = model(image)
            elif attack=='fgsm':
                output = model(image+delta)
            loss = criterion(output, target)
            if attack==None:
                data_min,data_max=transform_pertub(args.dataset,image,eps,bound=True)
                lb, ub, rs_loss, unstable_ratio,layerwise_loss, layerwise_ratio,layer_new_loss = compute_bound(model,data_min,
                                                                              data_max,gamma_pow=args.gamma_pow,
                                                                              stat_stability=True,return_nrs=True)

                for key in layer_new_loss:
                    layer_new_losses[key]+=layer_new_loss[key]


        output = output.float()
        loss = loss.float()
        losses.update(loss.item(), image.size(0))
        prec1 = accuracy(output.data, target)[0]
        top1.update(prec1.item(), image.size(0))

        if attack == None:
            rs_loss = rs_loss.float()
            rs_losses.update(rs_loss.item(), image.size(0))
            unstable_ratios.update(unstable_ratio,image.size(0))
            for j in range(len(layerwise_loss)):
                if len(layerwise_losses)<=j:
                    layerwise_losses.append(AverageMeter())
                layerwise_losses[j].update(layerwise_loss[j])
            for j in range(len(layerwise_ratio)):
                if len(layerwise_ratios)<=j:
                    layerwise_ratios.append(AverageMeter())
                layerwise_ratios[j].update(layerwise_ratio[j])



    if attack==None:
        logger.info('Test: \t'
            'Loss {loss.val:.4f} ({loss.avg:.4f}), rs_loss {rs_loss.val:.4f} ({rs_loss.avg:.4f})\t'
            'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                 loss=losses, rs_loss=rs_losses, top1=top1))
    elif attack=='fgsm':
        logger.info('Robust test: \t'
              'Loss {loss.val:.4f} ({loss.avg:.4f}), \t'
              'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
             loss=losses, top1=top1))

    if attack==None:
        return top1.avg, rs_losses.avg, unstable_ratios.avg, [x.avg for x in layerwise_losses], [x.avg for x in
                                                                                                 layerwise_ratios],layer_new_losses
    elif attack=='fgsm':
        return top1.avg

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='last.model'):
    filepath = os.path.join(save_path, "%02d"%pruning+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, "%02d"%pruning+'best.model'))
#
# def warmup_lr(epoch, step, optimizer, one_epoch_step):
#
#     overall_steps = args.warmup*one_epoch_step
#     current_steps = epoch*one_epoch_step + step
#
#     lr = args.lr * current_steps/overall_steps
#     lr = min(lr, args.lr)
#
#     for p in optimizer.param_groups:
#         p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    logger.info('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()


