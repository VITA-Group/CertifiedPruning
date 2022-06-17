import os
import argparse
from os.path import join

parser=argparse.ArgumentParser()

parser.add_argument('--file', type=str, help='must specify a txt result file')
parser.add_argument('--show_stage', action='store_true', help='whether to show pruning stage')
args=parser.parse_args()

root_dir=join(os.environ('HOME'),'saved_models')
exps=[join(root_dir,x) for x in os.listdir(root_dir)]

for exp in exps:
    log=join(exp,'train.log')
    if os.path.exists(log):
