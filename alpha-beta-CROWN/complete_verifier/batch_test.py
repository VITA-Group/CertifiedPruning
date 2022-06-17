import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--dir', type=str, help='must specify a checkpoint dir')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--compress', type=int, default=0)
parser.add_argument('--save', type=str, default='result1.txt')
parser.add_argument('--keyword', type=str, default='last')
parser.add_argument('--mode', type=str, default='crown-acc',choices=['crown-acc','verified-acc'])
parser.add_argument('--log_lines', type=int, default=2)
parser.add_argument('--config',default='exp_configs/cifar_model_deep.yaml',type=str)

args=parser.parse_args()

models=[]
for x in os.listdir(args.dir):
    if args.keyword in x:
        models.append(os.path.join(args.dir,x))

print('models:',models)
models=sorted(models)

with open(args.save,'a+') as fw:
    fw.write('\n')
    fw.write('\n')
    fw.write('=================>test dir: '+args.dir+'\n')

output=[]
print('models:',models)
for model in models:
    s=('python robustness_verifier.py --config '+ args.config +' --load '+model + ' --mode '+ args.mode + (' --compress' if args.compress else ''))
    if '8.255' in args.dir:
        s+=' --epsilon 0.031372549'
    print(s)
    fp = os.popen(s)
    output.extend(list(fp.readlines())[-args.log_lines:])
    with open(args.save,'a+') as fw:
        fw.write('#############checkpoint: '+model+'\n')
        for i in range(args.log_lines,0,-1):
            fw.write(output[-i])
    fp.close()

for x in output:
    print(x.strip())
