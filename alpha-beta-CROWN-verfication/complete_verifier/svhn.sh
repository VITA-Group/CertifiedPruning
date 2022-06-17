#! /bin/bash

source ~/.bashrc 
conda activate alpha-beta-crown

gpu=2
#save=result_tmp_fgsm.txt
save=result_svhn_fgsm_refill.txt
#keyword=final
keyword=11last
log_lines=8
config=exp_configs/cifar_model_deep_complete_svhn.yaml
mode=verified-acc
#log_lines=3
#config=exp_configs/cifar_model_deep.yaml
#mode=crown-acc

verification(){
    echo CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
    CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
}

#verification 0 svhn_cibp_l1unstrunct
#verification 0 svhn_cibp_l1unstrunct_nrs
#verification 0 svhn_fgsm_l1unstruct
#verification 0 svhn_fgsm_l1unstruct_rs
#verification 0 svhn_fgsm_l1unstruct_nrs
#verification 0 svhn_fgsm_hydra
verification 1 svhn_fgsm_slim
#verification 1 svhn_fgsm_snip
