#! /bin/bash

source ~/.bashrc 
conda activate alpha-beta-crown

gpu=0
save=result_fashion_cibp.txt
#keyword=final
keyword=last
log_lines=8
config=exp_configs/cifar_model_deep_complete.yaml
mode=verified-acc
#log_lines=3
#config=exp_configs/cifar_model_deep.yaml
#mode=crown-acc

verification(){
    echo CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
    CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
}

verification 0 cibp_l1unstrunct_rs
verification 1 cibp_slim
