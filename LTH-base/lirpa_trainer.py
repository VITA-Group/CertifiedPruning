from torch.nn import CrossEntropyLoss

import time
from utils import compute_slim_loss
from bound_prop import compute_lirpa_nrs,stat_instability,compute_fast_ibp_reg,ibp_init
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.eps_scheduler import AdaptiveScheduler, FixedScheduler, SmoothedScheduler
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter, get_spec_matrix

#code based on: https://github.com/KaidiXu/auto_LiRPA/blob/master/examples/vision/cifar_training.py
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

    return meter,opt_meter,return_dict,loss
