import warnings
warnings.filterwarnings("ignore")
import argparse
import random
from setproctitle import setproctitle
import time
import logging
from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np


import torch
import torch.optim as optim
from thop import profile
from torch.nn import CrossEntropyLoss

cpu_num = 2 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

from utils import *
from pruner import *
from bound_prop import compute_lirpa_nrs,stat_instability,compute_fast_ibp_reg,ibp_init
from lirpa_trainer import lirpa_train
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.eps_scheduler import AdaptiveScheduler, FixedScheduler, SmoothedScheduler
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import MultiAverageMeter, get_spec_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/cibp_random.yml', type=str, help='must specify a config file')

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default=None, help='verification mode, load path')
parser.add_argument("--compress", type=int, default=0, help='verification mode, whether to load compressed model')
parser.add_argument("--heat", type=str, default='linear', help='verification heatmap mode, linear or log')
parser.add_argument("--no_loss_fusion", action="store_true", help='without loss fusion, slower training mode')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN","FAST-IBP"], help='method  of bound analysis')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options for relu')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')

args=set_env(parser)
if args.prune_method=='hydra':
    import hydra
setproctitle(args.save_dir)

def lirpa_train(args, model, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust', loss_fusion=True, final_node_name=None, prune=False):
    num_class = 10
    meter = MultiAverageMeter()
    opt_meter= MultiAverageMeter()
    return_dict={}
    if train or prune:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    def get_exp_module(bounded_module):
        for _, node in bounded_module.named_modules():
            # Find the Exp neuron in computational graph
            if isinstance(node, BoundExp):
                return node
        return None

    exp_module = get_exp_module(model)

    def get_bound_loss(x=None, c=None):
        return_dict={}
        if loss_fusion:
            bound_lower, bound_upper = False, True
        else:
            bound_lower, bound_upper = True, False

        if bound_type == 'IBP' or bound_type=='FAST-IBP':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None, final_node_name=final_node_name, no_replicas=True)
        elif bound_type == 'CROWN':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method='backward',
                                          bound_lower=bound_lower, bound_upper=bound_upper)
        elif bound_type == 'CROWN-IBP':
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method='backward')  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            ilb, iub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None, final_node_name=final_node_name, no_replicas=True)

        if args.rs_reg or args.prune_method=='nrs':
            nrs_loss,nrs_dict,instab_dict=compute_lirpa_nrs(model,gamma_pow=args.gamma_pow)
            return_dict['nrs_loss']=nrs_loss
            return_dict['nrs_dict']=nrs_dict
            return_dict['instab_dict']=instab_dict
        with torch.no_grad():
            unstable_ratio,neuron_unstable_ratio,unstable_dict,neuron_med_sum,neuron_med_sqr_sum,pre_act_interval,pre_bn_interval,bn_neuron_unstable_ratio,bound_prod,bn_bound_prod=stat_instability(model)
        return_dict['unstable_ratio']=unstable_ratio
        return_dict['neuron_unstable_ratio']=neuron_unstable_ratio
        return_dict['bn_neuron_unstable_ratio']=bn_neuron_unstable_ratio
        return_dict['unstable_dict']=unstable_dict
        return_dict['neuron_med']=neuron_med_sum
        return_dict['neuron_med_sqr']=neuron_med_sqr_sum
        return_dict['pre_act_interval']=pre_act_interval
        return_dict['pre_bn_interval']=pre_bn_interval
        return_dict['bound_prod']=bound_prod
        return_dict['bn_bound_prod']=bn_bound_prod

        if bound_type == 'CROWN-IBP':
            factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
            if factor < 1e-50:
                lb, ub = ilb, iub
            else:
                clb, cub = model(method_opt="compute_bounds", IBP=False, C=c, method='backward',
                             bound_lower=bound_lower, bound_upper=bound_upper, final_node_name=final_node_name, no_replicas=True)
                if loss_fusion:
                    ub = cub * factor + iub * (1 - factor)
                else:
                    lb = clb * factor + ilb * (1 - factor)

        if loss_fusion:
            if isinstance(model, BoundDataParallel):
                max_input = model(get_property=True, node_class=BoundExp, att_name='max_input')
            else:
                max_input = exp_module.max_input
            return None, torch.mean(torch.log(ub) + max_input),return_dict
        else:
            # Pad zero at the beginning for each example, and use fake label '0' for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            return lb, robust_ce, return_dict

    for i, (data, labels) in enumerate(loader):
        if train:
            opt.zero_grad()

        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-50:
            batch_method = "natural"
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels = data.cuda(), labels.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)
        if loss_fusion:
            if batch_method == 'natural' or not train:
                output = model(x, labels)  # , disable_multi_gpu=True
                regular_ce = torch.mean(torch.log(output))
            else:
                model(x, labels)
                regular_ce = torch.tensor(0., device=data.device)
            meter.update('CE', regular_ce.item(), x.size(0))
            x = (x, labels)
            c = None
        else:
            # Generate speicification matrix (when loss fusion is not used).
            c = get_spec_matrix(data, labels, num_class)
            x = (x,) if final_node_name is None else (x, labels)
            output = model(x, final_node_name=final_node_name)
            regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
            meter.update('CE', regular_ce.item(), x[0].size(0))
            meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).item() / x[0].size(0), x[0].size(0))

        if batch_method == 'robust':
            lb, robust_ce, opt_dict = get_bound_loss(x=x, c=c)
            loss = robust_ce
            meter.update('Unstable', opt_dict['unstable_ratio'].item(), data.size(0))
            opt_meter.update('neuron_unstable',opt_dict['neuron_unstable_ratio'],data.size(0))
            opt_meter.update('bn_neuron_unstable',opt_dict['bn_neuron_unstable_ratio'],data.size(0))
            opt_meter.update('neuron_med', opt_dict['neuron_med'], data.size(0), opt_dict['neuron_med_sqr'])
            opt_meter.update('pre_act_interval',opt_dict['pre_act_interval'],data.size(0))
            opt_meter.update('pre_bn_interval',opt_dict['pre_bn_interval'],data.size(0))
            opt_meter.update('bound_prod',opt_dict['bound_prod'],data.size(0))
            opt_meter.update('bn_bound_prod',opt_dict['bn_bound_prod'],data.size(0))

            if bound_type=='FAST-IBP':
                loss+=compute_fast_ibp_reg(model,eps_scheduler)

            if args.rs_reg:
                loss += args.rs_weight * opt_dict['nrs_loss']
            if args.prune_method=='nrs':
                # for key in opt_dict['nrs_dict']:
                for key in opt_dict['instab_dict']:
                    opt_meter.update(key, opt_dict['instab_dict'][key], data.size(0))

        elif batch_method == 'natural':
            loss = regular_ce

        if prune:
            return loss

        if args.slim_weight>0:
            slim_loss=compute_slim_loss(model)
            loss+=args.slim_weight*slim_loss


        if train:
            loss.backward()

            if args.clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                meter.update('grad_norm', grad_norm.item())

            if isinstance(eps_scheduler, AdaptiveScheduler):
                eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))

        # check gradient
        # for n, p in model.named_parameters():
        #     if p.grad is None:
        #         print('gradient for layer {} is NULL!!!'.format(n))
        #     else:
        #         print('gradient for layer {} is not null'.format(n))
        #         print(p.grad.flatten()[:8])
        #
        # sys.exit()

        if batch_method != 'natural':
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            if not loss_fusion:
                # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
                # If any margin is < 0 this example is counted as an error
                meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)

        if (i + 1) % 50 == 0 and train:
            logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))

    logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))

    if batch_method=='robust':
        return_dict['unstable_ratio']=meter.avg('Unstable')
        if args.prune_method=='nrs':
            return_dict['instab_dict']={}
            for key in opt_dict['instab_dict'].keys():
                return_dict['instab_dict'][key]=opt_meter.avg(key)

    return meter,opt_meter,return_dict

#code based on: https://github.com/KaidiXu/auto_LiRPA/blob/master/examples/vision/cifar_training.py
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    epoch = 0
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     epoch, state_dict = checkpoint['epoch'], checkpoint['state_dict']
    #     opt_state = None
    #     try:
    #         opt_state = checkpoint['optimizer']
    #     except KeyError:
    #         print('no opt_state found')
    #     for k, v in state_dict.items():
    #         assert torch.isnan(v).any().cpu().numpy() == 0 and torch.isinf(v).any().cpu().numpy() == 0
    #     model_ori.load_state_dict(state_dict)
    #     logger.info('Checkpoint loaded: {}'.format(args.resume))

    ## Step 1-2:Initial original model as usual, Prepare dataset as usual
    model_ori, train_loader,  test_loader = setup_model_dataset(args)
    if args.verify:
        load_path=os.path.join(args.save_dir,args.load)
        model_ori=load_model(model_ori,load_path,args,args.compress)
        print(model_ori)

    if args.dataset == 'mnist':
        dummy_input = torch.randn(2, 1, 28, 28)
    elif args.dataset == 'FashionMNIST':
        dummy_input = torch.randn(2, 1, 28, 28)
    elif args.dataset == 'cifar10':
        dummy_input = torch.randn(2, 3, 32, 32)
    elif args.dataset == 'svhn':
        dummy_input = torch.randn(2, 3, 32, 32)
    norm = float(args.norm)

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    logger.info(str(model_ori))
    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)

    if args.bound_type=='FAST-IBP':
        ibp_init(model_ori,model)

    final_name1 = model.final_name
    model_loss = BoundedModule(CrossEntropyWrapper(model_ori), (dummy_input, torch.zeros(1, dtype=torch.long)),
                               bound_opts={'relu': args.bound_opts, 'loss_fusion': True}, device=args.device)
    # after CrossEntropyWrapper, the final name will change because of one additional input node in CrossEntropyWrapper
    final_name2 = model_loss._modules[final_name1].output_name[0]
    assert type(model._modules[final_name1]) == type(model_loss._modules[final_name2])
    if args.no_loss_fusion:
        model_loss = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
        final_name2 = None
    model_loss = BoundDataParallel(model_loss)
    # if args.prune_method=='hydra':
    #     model_hydra=copy.deepcopy(model_loss)
    #     for name,m in model_hydra.named_modules():
    #         if isinstance(m,BoundConv):
    #             m.inputs[1].set_hydra_mode()

    macs, params = profile(model_ori, (dummy_input.cuda(),))
    logger.info('macs: {}, params: {}'.format(macs, params))

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    logger.info(str(model_ori))

    # skip epochs
    # if epoch > 0:
    #     epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
    #     eps_scheduler.set_epoch_length(epoch_length)
    #     eps_scheduler.train()
    #     for i in range(epoch):
    #         lr_scheduler.step()
    #         eps_scheduler.step_epoch(verbose=True)
    #         for j in range(epoch_length):
    #             eps_scheduler.step_batch()
    #     logger.info('resume from eps={:.12f}'.format(eps_scheduler.get_eps()))

    # if args.resume:
    #     if opt_state:
    #         opt.load_state_dict(opt_state)
    #         logger.info('resume opt_state')

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            m,opt_m,opt_dict=lirpa_train(args,model, 1, test_loader, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=None)
            instability=opt_m.avg('neuron_unstable')
            sensitivity=opt_m.std('neuron_med')
            x=(1*instability).tolist()
            y=(1*sensitivity).tolist()
            bins=100
            xrange=[0,0.10]
            yrange=[0,1.75]
            if args.heat=='linear':
                hist, xbins, ybins, im = plt.hist2d(x, y, range=[xrange,yrange],bins=bins,norm=mpl.colors.Normalize(vmax=100,clip=True))
            elif args.heat=='log':
                hist, xbins, ybins, im = plt.hist2d(x, y, range=[xrange,yrange],bins=bins,norm=mpl.colors.LogNorm(vmax=1000,clip=True))
            plt.xlabel('unstable ratio')
            plt.ylabel('std deviation')
            plt.title('active neurons: {:.2f}%'.format(100*(1-hist[0][0].item()/len(x))))
            model_name = args.load[:-10]
            plt.savefig(os.path.join(args.save_dir, args.heat+'_heat_'+model_name+'.png'))
            plt.close()
            print('avg unstable ratio:', instability.mean().item())
            print('avg bn unstable ratio:', opt_m.avg('bn_neuron_unstable').mean().item())
            print('instability/sensitivity:',(instability/(1e-5+sensitivity)).mean().item())
            print('clean acc:',(1-m.avg('Err')))
            print('pre-bn bound interval:',opt_m.avg('pre_bn_interval').mean().item())
            print('pre-act bound interval:',opt_m.avg('pre_act_interval').mean().item())
            print('pre-act bound product:',opt_m.avg('bound_prod').mean().item())
            print('pre-bn bound product:',opt_m.avg('bn_bound_prod').mean().item())

    else:
        err_result=[]
        verr_result=[]
        sparsities=[]
        channel_sparsities=[]

        initialization=model_loss.module.state_dict()
        for state in range(0, args.prune_times):
            logger.info('******************************************')
            logger.info('pruning state {}'.format(state))
            logger.info('******************************************')

            sparsity,channel_sparsity,_=check_sparsity(model_loss)
            sparsities.append(sparsity / 100)
            channel_sparsities.append(channel_sparsity / 100)
            plt.ylim(0, 1)
            plt.plot(channel_sparsities, label='channels')
            plt.plot(sparsities, label='overall weights')
            plt.ylabel('REMAINING WEIGHT')
            plt.xlabel('prune_time@%.2f' % args.rate)
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, 'sparsity.png'))
            plt.close()


            if args.prune_type!='finetune' or state==0:
                opt = optim.Adam(model_loss.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_decay_milestones,gamma=0.1)
                eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)

            if args.one_shot:
                logger.info('score-based pruning')
                prune_model_score_based(args,model_loss, 1 - args.rate, 'cibp', args.prune_method, True,
                                        dataloader=train_loader,
                                        optimizer=opt, eps_scheduler=FixedScheduler(args.eps), norm=norm,
                                        bound_type=args.bound_type, loss_fusion=not args.no_loss_fusion)
                check_sparsity(model_loss)

            timer = 0.0
            best_verr = 1
            best_err = 1
            best_opt_dict=None
            unstable_ratios=[]
            # with torch.autograd.detect_anomaly():
            for t in range(epoch + 1, args.epochs+1):
                logger.info("Epoch {}, learning rate {}".format(t, lr_scheduler.get_last_lr()))
                start_time = time.time()
                lirpa_train(args,model_loss, t, train_loader, eps_scheduler, norm, True, opt, args.bound_type, loss_fusion=not args.no_loss_fusion)
                lr_scheduler.step()
                epoch_time = time.time() - start_time
                timer += epoch_time
                logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

                logger.info("Evaluating...")
                torch.cuda.empty_cache()


                # remove 'model.' in state_dict for CrossEntropyWrapper
                state_dict_loss = model_loss.state_dict()
                state_dict = {}
                if not args.no_loss_fusion:
                    for name in state_dict_loss:
                        assert (name.startswith('model.'))
                        state_dict[name[6:]] = state_dict_loss[name]
                else:
                    state_dict = state_dict_loss

                if state == 0:
                    if (epoch + 1) == args.rewind_epoch:
                        torch.save(state_dict,
                                   os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch + 1)))
                        if args.prune_type == 'rewind_lt':
                            initialization = model_loss.module.state_dict()

                with torch.no_grad():
                    if t <= int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                        m, opt_m, opt_dict = lirpa_train(args,model_loss, t, test_loader, eps_scheduler, norm, False, None, 'IBP',
                                           loss_fusion=False, final_node_name=final_name2)
                    m, opt_m, opt_dict = lirpa_train(args,model_loss, t, test_loader, FixedScheduler(args.eps), norm, False, None, 'IBP',
                                        loss_fusion=False, final_node_name=final_name2)

                save_dict = {'state_dict': state_dict, 'epoch': t, 'optimizer': opt.state_dict()}
                current_err = m.avg('Err')
                current_verr = m.avg('Verified_Err')
                if current_verr <= best_verr:
                    best_err = current_err
                    best_verr = current_verr
                    best_opt_dict=opt_dict
                    torch.save(save_dict, os.path.join(args.save_dir,"%02d"%state+"best.model"))
                torch.save(save_dict, os.path.join(args.save_dir,"%02d"%state+"last.model"))
            torch.cuda.empty_cache()


            if args.one_shot:
                break
            # pruning and rewind
            remain_ratio = math.pow(1 - args.rate, state + 1)
            if args.prune_method == 'random':
                logger.info('random pruning')
                prune_model_unstructured(model_loss, args.rate,'random')
            elif args.prune_method == 'l1unstruct':
                logger.info('L1 unstructrued pruning')
                prune_model_unstructured(model_loss, args.rate,'l1unstruct')
            elif args.prune_method == 'snip' or args.prune_method == 'taylor1ScorerAbs':
                logger.info('score-based pruning')
                prune_model_score_based(args,model_loss, remain_ratio, 'cibp', args.prune_method, False, dataloader=train_loader,
                                        optimizer=opt, eps_scheduler=FixedScheduler(args.eps), norm=norm,
                                        bound_type=args.bound_type, loss_fusion=not args.no_loss_fusion)
            # elif args.prune_method == 'l1-channel':
            #     print('L1 channel pruning')
            #     prune_model_ln_structured(model_loss, args.rate)
            elif args.prune_method=='refill':
                logger.info('IMP Refilling...')
                prune_model_imp_refill(model_loss,args.imp_rate,remain_ratio,local=args.layerwise)
            elif args.prune_method == 'slim':
                print('network slimming...')
                prune_model_structured(model_loss, remain_ratio, mode='slim',local=args.layerwise)
            elif args.prune_method == 'nrs':
                print('NRSLoss pruning')
                prune_model_structured(model_loss, remain_ratio, mode='nrs', local=args.layerwise, loss=best_opt_dict['instab_dict'])
            elif args.prune_method == 'hydra':
                print('Hydra robustness-based pruning')
                hydra.prune_model_hydra(args, model_loss, remain_ratio, train_loader)
            else:
                raise NotImplementedError('Pruning method not implemented!')


            # weight rewinding
            if args.prune_type!='finetune':
                check_sparsity(model_loss)
                current_mask = extract_mask(model_loss)
                remove_prune(model_loss)
                model_loss.module.load_state_dict(initialization)
                prune_model_custom(model_loss, current_mask)

            err_result.append(best_err)
            verr_result.append(best_verr)
            unstable_ratios.append(best_opt_dict['unstable_ratio'])

            plt.ylim(0, 1)
            plt.plot(err_result, label='Std Err')
            plt.plot(verr_result, label='Verified Err')
            plt.ylabel('Err Rate')
            plt.xlabel('prune_time@%.2f' % args.rate)
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, 'ErrRate.png'))
            plt.close()


            plt.ylim(0, 1)
            plt.plot(unstable_ratios, label='Unstable Ratio')
            plt.ylabel('Unstable Ratio ')
            plt.xlabel('prune_time@%.2f' % args.rate)
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, 'Instability.png'))
            plt.close()

        logger.info('Err result:')
        logger.info(err_result)
        logger.info('Verified err result:')
        logger.info(verr_result)

if __name__ == "__main__":
    logger.info(args)
    main(args)
