import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--file', type=str, help='must specify a txt result file')
parser.add_argument('--show_stage', action='store_true', help='whether to show pruning stage')
parser.add_argument('--mode', default='ver', choices=['ver','pgd','time'])
args=parser.parse_args()

s=''
with open(args.file,'r') as fp: 
    l=list(fp.readlines())
    for i in range(len(l)):
        line=l[i]
        line=line.strip()
        if '===>' in line:
            s='stage\t'+line.split('/')[-1]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
        elif '####check' in line:
            s=line.split('/')[-1][:2]
            if (args.mode=='ver' and (not l[i+3].startswith('final verifi')) and (not l[i+4].startswith('final verifi'))) or (args.mode=='pgd' and (not l[i+2].startswith('attack_success rate'))) or (args.mode=='time' and (not 'including attack' in l[i+6])):
                if args.show_stage:
                    print(s+'\t0')
                else:
                    print('#')
        elif args.mode=='ver' and ('final verifi' in line):
            s+='\t'+line.split(' ')[-1].split('%')[0]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
        elif args.mode=='ver' and ('final verifi' in line):
            s+='\t'+line.split(' ')[-1].split('%')[0]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
        elif args.mode=='pgd' and ('attack_success rate' in line):
            s+='\t'+str(100*float(line.split(' ')[-1]))
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
        elif args.mode=='time' and ('including attack' in line):
            s+='\t'+line.split(' ')[-1]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
        elif 'CROWN clean' in line:
            s+='\t'+line.split('verified acc: ')[-1].split('%')[0]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
