import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--file', type=str, help='must specify a txt result file')
parser.add_argument('--show_stage', action='store_true', help='whether to show pruning stage')
parser.add_argument('--keyword', type=str, nargs='+')
args=parser.parse_args()
args.keyword=' '.join(args.keyword)

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
        elif '#######' in line:
            s=line.split('/')[-1][:2]
#            if i+3<len(l) and (not 'incomplete' in args.file) and (not l[i+3].startswith('final verifi')):
#                if args.show_stage:
#                    print(s+'\t0')
#                else:
#                    print(0)
        elif args.keyword in line:
            s+='\t'+line.split(': ')[-1]
            if args.show_stage:
                print(s)
            else:
                print(s.split('\t')[-1])
            s=''
