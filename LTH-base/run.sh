#! /bin/bash
source ~/.bashrc
conda activate alpha-beta-crown

gpu=0
log_lines=8
keyword=last
heat=linear

stat_and_plot(){
    echo CUDA_VISIBLE_DEVICES=$gpu python plot_neuron_stat.py   --log_lines $log_lines --keyword $keyword --heat $heat --compress $1 --dir $HOME/saved_models/$2
    CUDA_VISIBLE_DEVICES=$gpu python plot_neuron_stat.py  --log_lines $log_lines --keyword $keyword --heat $heat --compress $1 --dir $HOME/saved_models/$2
}

#stat_and_plot 0 cibp_random
#stat_and_plot 0 cibp_l1unstruct
#stat_and_plot 0 cibp_l1unstruct_nrs
#stat_and_plot 0 cibp_l1unstruct_rs
#stat_and_plot 0 cibp_snip_decay
#stat_and_plot 0 cibp_taylor1_decay
#stat_and_plot 0 cibp_l1unstruct_nrs_seed200
#stat_and_plot 0 cibp_l1unstruct_nrs_seed300
#stat_and_plot 0 cibp_l1unstruct_nrs_gammaDetach
#stat_and_plot 1 cibp_slim_decay
#stat_and_plot 1 cibp_nrs
#stat_and_plot 1 cibp_refill_decay
#
#stat_and_plot 0 cibp_l1unstruct_8.255
#stat_and_plot 0 cibp_l1unstruct_decay0
#stat_and_plot 0 cibp_l1unstruct_nrs_gammaDetach_seed200
#stat_and_plot 0 cibp_l1unstruct_nrs_gammaDetach_seed300
#stat_and_plot 0 cibp_l1unstruct_seed200
#stat_and_plot 0 cibp_l1unstruct_seed300
#stat_and_plot 1 cibp_slim_withloss
#stat_and_plot 0 fgsm_l1unstruct
#stat_and_plot 0 fgsm_l1unstruct_nrs
#stat_and_plot 0 fgsm_l1unstruct_nrs_decay5e-4
#stat_and_plot 0 fgsm_l1unstruct_rs
#stat_and_plot 0 fgsm_random
#stat_and_plot 1 fgsm_refill
#stat_and_plot 1 fgsm_slim
#stat_and_plot 0 fgsm_snip
#stat_and_plot 0 fgsm_snip_1shot
#stat_and_plot 0 fgsm_taylor1ScorerAbs
#stat_and_plot 0 fgsm_taylor1ScorerAbs_1shot
#stat_and_plot 0 cibp_random_8.255
#stat_and_plot 0 cibp_l1unstruct_8.255
#stat_and_plot 0 cibp_l1unstruct_nrs_8.255_gammaDetach
#stat_and_plot 0 cibp_l1unstruct_rs_8.255
stat_and_plot 0 fgsm_random
stat_and_plot 0 fgsm_l1unstruct
stat_and_plot 0 fgsm_l1unstruct_nrs
stat_and_plot 0 fgsm_snip
