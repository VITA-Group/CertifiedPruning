import os
import warnings
import warnings
warnings.filterwarnings("ignore")
import argparse
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()

parser.add_argument('--dir', type=str, help='must specify a checkpoint dir')
parser.add_argument('--keyword', type=str, default='last')
parser.add_argument('--compress',type=int,default=0)
parser.add_argument('--heat',type=str,default='linear')
parser.add_argument('--log_lines',type=int,default=8)
parser.add_argument('--rate',type=float,default=0.2,help='pruning rate (for plotting)')


args=parser.parse_args()

models=[]
for x in os.listdir(args.dir):
    if args.keyword in x:
        models.append(x)

models=sorted(models)

print('=============>',args.dir)

output=[]
unstable=[]
bn_unstable=[]
insta_sensi=[]
clean_acc=[]
pre_bn=[]
pre_act=[]
bound_prod=[]
bn_bound_prod=[]
for model in models:
    cmd='python crown_ibp.py --verify --save_dir '+ args.dir +' --load '+model + ' --compress ' +str(args.compress)+' --heat '+args.heat
    if '8.255' in args.dir:
        cmd+=' --eps 8/255'
    fp = os.popen(cmd)
    print('###########', os.path.join(args.dir, model))
    for line in list(fp.readlines())[-args.log_lines:]:
        line=line.strip()
        print(line)
        if 'avg unstable ratio' in line:
            unstable.append(float(line.split(':')[-1]))
        if 'avg bn unstable ratio' in line:
            bn_unstable.append(float(line.split(':')[-1]))
        elif 'sensitivity' in line:
            insta_sensi.append(float(line.split(':')[-1]))
        elif 'clean acc' in line:
            clean_acc.append(float(line.split(':')[-1]))
        elif 'pre-bn bound interval' in line:
            pre_bn.append(float(line.split(':')[-1]))
        elif 'pre-act bound interval' in line:
            pre_act.append(float(line.split(':')[-1]))
        elif 'pre-act bound product' in line:
            bound_prod.append(float(line.split(':')[-1]))
        elif 'pre-bn bound product' in line:
            bn_bound_prod.append(float(line.split(':')[-1]))

if 'fgsm' not in args.dir:
    plt.ylim(0, 0.04)
plt.plot(unstable, label='unstable_ratio')
plt.plot(bn_unstable, label='bn_unstable_ratio')
plt.plot(insta_sensi, label='instability/sensitivity')
plt.xlabel('prune_time@%.2f' % args.rate)
plt.legend()
plt.savefig(os.path.join(args.dir, 'neuron_insta_sensi.png'))
plt.close()


plt.ylim(0,1)
plt.plot(clean_acc, label='clean_acc')
plt.xlabel('prune_time@%.2f' % args.rate)
plt.legend()
plt.savefig(os.path.join(args.dir, 'clean_acc.png'))
plt.close()

if 'fgsm' not in args.dir:
    plt.ylim(0,0.3)
plt.plot(pre_act, label='pre-act interval')
plt.plot(pre_bn, label='pre-bn interval')
plt.xlabel('prune_time@%.2f' % args.rate)
plt.legend()
plt.savefig(os.path.join(args.dir, 'neuron_interval.png'))
plt.close()

plt.plot(bound_prod, label='- pre-act bound product')
plt.plot(bn_bound_prod, label='- pre-bn bound product')
plt.xlabel('prune_time@%.2f' % args.rate)
plt.legend()
plt.savefig(os.path.join(args.dir, 'bound_prod.png'))
plt.close()


os.system('python makegif.py '+args.dir+' '+args.heat+'_heat')

print(unstable)
