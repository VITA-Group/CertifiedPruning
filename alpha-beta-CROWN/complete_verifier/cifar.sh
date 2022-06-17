#! /bin/bash

source ~/.bashrc 
conda activate alpha-beta-crown

gpu=0
save=result_fashion_cibp.txt
#keyword=final
keyword=last
log_lines=8
config=exp_configs/cifar_model_deep_complete_FashionMNIST.yaml
mode=verified-acc
#log_lines=3
#config=exp_configs/cifar_model_deep.yaml
#mode=crown-acc

verification(){
    echo CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
    CUDA_VISIBLE_DEVICES=$gpu python batch_test.py --save $save --config $config --log_lines $log_lines --mode $mode --keyword $keyword --compress $1 --dir $HOME/saved_models/$2
}

#verification 0 fashion_cibp_l1unstrunct
verification 0 fashion_cibp_l1unstrunct_rs
verification 0 fashion_cibp_l1unstrunct_nrs
verification 1 fashion_cibp_slim
verification 1 fashion_cibp_refill
verification 1 fashion_cibp_nrs
#verification 0 cibp_random
#verification 0 cibp_l1unstruct
#verification 0 cibp_l1unstruct_nrs
#verification 0 cibp_l1unstruct_rs
#verification 0 cibp_hydra
#verification 1 cibp_slim
#verification 0 cibp_snip
#verification 0 cibp_taylor1
#verification 1 cibp_nrs
#verification 1 cibp_slim
#
#verification 0 fgsm_random
#verification 0 fgsm_snip
#verification 0 fgsm_taylor1
#
#
#verification 0 cibp_snip_1shot
#verification 0 cibp_taylor1_1shot
#verification 0 fgsm_snip
#verification 0 fgsm_taylor1ScorerAbs_1shot

#verification 1 cibp_slim_withloss
#verification 1 cibp_slim_layerwise

#verification 1 cibp_nrs
#verification 0 fgsm_random

#verification 1 cibp_refill
#verification 0 cibp_l1unstruct
#verification 0 cibp_snip
#verification 0 cibp_taylor1
#verification 1 cibp_refill_layerwise

#verification 0 cibp_random
#imcomplete
#verification 1 fgsm_refill
#verification 0 fgsm_l1unstruct
#verification 1 fgsm_slim

#verification 1 cibp_slim_decay
#verification 1 cibp_refill_decay
#verification 0 cibp_taylor1_decay
#verification 0 cibp_l1unstruct_nrs
#verification 0 cibp_random_decay
#verification 0 cibp_snip_decay
#verification 0 cibp_l1unstruct_rerun
#verification 0 cibp_l1unstruct

#verification 0 cibp_random_8.255
#verification 0 cibp_l1unstruct_rerun2
#verification 0 cibp_l1unstruct_8.255
#verification 0 cibp_l1unstruct_decay0
#verification 1 cibp_nrs
#verification 0 cibp_l1unstruct_nrs_seed200
#verification 0 cibp_l1unstruct_nrs_seed300
#verification 0 cibp_l1unstruct_rs
#verification 0 cibp_l1unstruct_nrs_rsweight0.01
#verification 0 cibp_l1unstruct_nrs_rsweight0.0001
#verification 0 cibp_l1unstruct_seed200
#verification 0 cibp_l1unstruct_seed300

#verification 1 cibp_nrs_layerwise

#verification 0 fgsm_l1unstruct_nrs_decay5e-4
#verification 1 fgsm_slim
#verification 0 fgsm_l1unstruct_nrs
#verification 0 fgsm_l1unstruct_rs
#verification 1 fgsm_refill
#verification 0 cibp_random_8.255
#verification 0 cibp_l1unstruct_8.255
#verification 0 cibp_l1unstruct_rs_8.255
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach
#verification 0 cibp_l1unstruct_nrs_gammaDetach
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed200
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed300
#verification 0 cibp_l1unstruct_nrs_gammaDetach_decay0.00001
#verification 0 cibp_l1unstruct_rs_decay1e-5

#verification 1 fgsm_nrs_slim
#verification 0 fgsm_snip
#verification 0 fgsm_taylor1ScorerAbs

#verification 0 fgsm_l1unstruct_rs_decay5e-4
#verification 0 fgsm_random_earlyWeight
#verification 0 fgsm_l1unstruct_earlyWeight
#verification 0 fgsm_l1unstruct_nrs_earlyWeight
#verification 0 fgsm_l1unstruct_earlyWeight20
#verification 0 cibp_l1unstruct_seed400
#verification 0 cibp_l1unstruct_seed500
#verification 0 cibp_l1unstruct_nrs_seed400
#verification 0 cibp_l1unstruct_nrs_seed500
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed400
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed500
#verification 0 cibp_taylor1_seed400
#verification 0 cibp_taylor1_seed500
#verification 0 cibp_snip_seed400
#verification 0 cibp_snip_seed500
#verification 1 cibp_refill_seed400
#verification 1 cibp_refill_seed500
#verification 1 cibp_slim_seed200
#verification 1 cibp_slim_seed300
#verification 0 cibp_random_seed200
#verification 0 cibp_random_seed300
#verification 0 cibp_snip_seed200
#verification 0 cibp_snip_seed300
#verification 0 cibp_taylor1_seed200
#verification 0 cibp_taylor1_seed300
#verification 1 cibp_refill_seed200
#verification 1 cibp_refill_seed300
#verification 0 cibp_l1unstruct_nrs_detach_decay
#verification 0 cibp_l1unstruct_nrs_detach_decay_seed200
#verification 0 cibp_l1unstruct_nrs_detach_decay_seed300
#verification 0 cibp_l1unstruct_nrs_detach_decay_seed400
#verification 0 cibp_l1unstruct_nrs_detach_decay_seed500
#verification 0 cibp_l1unstruct_nrs_decay
#verification 0 cibp_l1unstruct_nrs_decay_seed200
#verification 0 cibp_l1unstruct_nrs_decay_seed300
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach_rerun
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach_seed200
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach_seed300
#verification 0 cibp_l1unstruct_nrs_rerun
#verification 0 cibp_l1unstruct_nrs_rerun_seed200
#verification 0 cibp_l1unstruct_nrs_rerun_seed300
#verification 0 cibp_l1unstruct_nrs_gammaDetach
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed200
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed300
#verification 0 fgsm_l1unstruct_8.255
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach_seed200
#verification 0 cibp_l1unstruct_nrs_8.255_gammaDetach_seed300
#verification 0 cibp_l1unstruct_nrs_gammaDetach_seed400
#verification 0 cibp_l1unstruct_nrs_gammaDetach_decay
#verification 0 cibp_l1unstruct_finetune
#verification 0 cibp_l1unstruct_nrs_detach_seed400_renew
#verification 0 cibp_l1unstruct_nrs_detach_seed400_decay_renew
#verification 0 cibp_hydra
#verification 0 cibp_hydra_seed200
#verification 0 cibp_hydra_seed300
#verification 0 cibp_hydra_seed400
#verification 0 cibp_hydra_seed500
#verification 0 cibp_l1unstruct_rs_8.255_seed200
#verification 0 cibp_l1unstruct_rs_8.255_seed300
#verification 0 cibp_l1unstruct
#verification 0 cibp_snip
#verification 0 cibp_taylor1
#verification 0 cibp_hydra
#verification 0 cibp_l1unstruct_rs
#verification 0 cibp_l1unstruct_nrs_gammaDetach
#verification 1 cibp_refill
#verification 1 cibp_slim
#verification 0 fgsm_l1unstruct
#verification 1 fgsm_refill
#verification 1 fgsm_slim
#verification 0 fgsm_hydra
#verification 0 fgsm_l1unstruct_nrs
#verification 0 fgsm_l1unstruct_rs
#verification 0 cibp_l1unstrunct_nrs_rebut
#verification 0 cibp_l1unstrunct_nrs_rebut_seed200
#verification 0 cibp_l1unstrunct_nrs_rebut_seed300
#verification 0 cibp_l1unstrunct_nrs_rebut_seed400
#verification 0 cibp_l1unstrunct_nrs_rebut_seed500
#verification 0 cibp_l1unstrunct_nrs_rebut_slim0.0001
#verification 0 cibp_l1unstrunct_nrs_rebut_slim0.0001_seed_200
#verification 0 cibp_l1unstrunct_nrs_rebut_slim0.0001_seed_300
#verification 0 cibp_l1unstrunct_nrs_rebut
