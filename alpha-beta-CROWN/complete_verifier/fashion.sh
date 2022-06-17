#! /bin/bash

source ~/.bashrc 
conda activate alpha-beta-crown

gpu=3
save=result_fashion_fgsm_sniptp_norm1.txt
#keyword=final
keyword=15last
log_lines=8
config=exp_configs/cifar_model_deep_complete_FashionMNIST.yaml
mode=verified-acc
#log_lines=3
#config=exp_configs/cifar_model_deep_FashionMNIST.yaml
#mode=crown-acc

verification(){
    echo CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
    CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
}

#verification 0 fashion_fgsm_l1unstruct
#verification 0 fashion_fgsm_hydra
#verification 0 fashion_fgsm_l1unstruct_rs
#verification 0 fashion_fgsm_l1unstruct_nrs
#
#verification 1 fashion_fgsm_slim
#verification 1 fashion_fgsm_refill
#verification 1 fashion_fgsm_nrs
#verification 0 fashion_cibp_snip
#verification 0 fashion_cibp_taylor1
#verification 0 fashion_fgsm_l1unstruct
#verification 0 fashion_fgsm_l1unstruct_rs
#verification 0 fashion_fgsm_l1unstruct_nrs
#verification 0 fashion_fgsm_hydra
verification 0 fashion_fgsm_snip
verification 0 fashion_fgsm_taylor1
#verification 1 fashion_fgsm_slim
#verification 1 fashion_fgsm_refill

