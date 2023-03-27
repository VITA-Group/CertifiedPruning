# CertifiedPruning
Code for paper [(TMLR) Can pruning improve certified robustness of deep neural networks?](https://arxiv.org/abs/2206.07311)

## Overview

With the rapid development of deep learning, the sizes of neural networks become larger and larger so that the training and inference often overwhelm the hardware resources. Given the fact that neural networks are often over-parameterized, one effective way to reduce such computational overhead is neural network pruning, by removing redundant parameters from trained neural networks. It has been recently observed that pruning can not only reduce computational overhead but also can improve empirical robustness of deep neural networks (NNs), potentially owing to removing spurious correlations while preserving the predictive accuracies. This paper for the first time demonstrates that pruning can generally improve certified robustness for ReLU-based NNs under the complete verification setting. Using the popular Branch-and-Bound (BaB) framework, we find that pruning can enhance the estimated bound tightness of certified robustness verification, by alleviating linear relaxation and sub-domain split problems. We empirically verify our findings with off-the-shelf pruning methods and further present a new stability-based pruning method tailored for reducing neuron instability, that outperforms existing pruning methods in enhancing certified robustness. Our experiments show that by appropriately pruning an NN, its certified accuracy can be boosted up to 8.2% under standard training, and up to 24.5% under adversarial training on the CIFAR10 dataset. We additionally observe the existence of certified lottery tickets that can match both standard and certified robust accuracies of the original dense models across different datasets. Our findings offer a new angle to study the intriguing interaction between sparsity and robustness, i.e. interpreting the interaction of sparsity and certified robustness via neuron stability.

## The proposed NRSLoss regularizer
<img width="843" alt="image" src="https://user-images.githubusercontent.com/15967092/174507806-21db873a-a4c7-465c-9ea5-07b6b0fae3e5.png">

## Performance

We observe consistent improvement on certified robustness of our tested pruning methods.

<img width="899" alt="image" src="https://user-images.githubusercontent.com/15967092/174507920-430929f0-d1f5-4d9e-9a1b-6a20c76aea6c.png">

We also tested the SOTA certified training methods, results show consistent improvement by different pruning methods. (Note that FastIBP trains 4x slower than Auto-LiRPA)

<img width="681" alt="image" src="https://user-images.githubusercontent.com/15967092/174508018-c238d0a3-6fed-49dc-9396-5f5850b525ec.png">

## Usage
Note: we choose alpha-beta-CROWN as the verification framework, hence our code is largely based on their [source repo](https://github.com/huanzhang12/alpha-beta-CROWN)

We recommend installing the alpha-beta-CROWN virtual environment following [this instruction](https://github.com/huanzhang12/alpha-beta-CROWN#installation-and-setup) via conda first, and then install additive dependencies for LTH-training.

The code is divided to training (`LTH-training` subdir) and verification (`alpha-beta-CROWN-verification` subdir) parts.
Please go to the subdir for instructions on training and verification.

Note:
Before training: 
```
mkdir ~/saved_models
```

After training, the checkpoints are saved under `~/saved_models/<args.save_dir>` folder.

For verification, specify the same <args.save_dir> folder in `cifar.sh/svhn.sh/fashion.sh` for batch verification of all subnetworks produced by iterative pruning.
```
#example in cifar.sh
cibp_l1unstruct 0 #0 means models not compressed for unstructured pruning methods
cibp_slim 1 #1 means models will be compressed for structured pruning methods
```

## Citation
If your find this code helpful, please cite the following:

```
@article{li2022can,
  title={Can pruning improve certified robustness of neural networks?},
  author={Li, Zhangheng and Chen, Tianlong and Li, Linyi and Li, Bo and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2206.07311},
  year={2022}
}
```
