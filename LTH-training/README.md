# Training code to find certified lottery tickets



### Environment:

​	pytorch

​	torchvision

​	thop

​	matplotlib


### Usage:

##### FGSM training

```
python fgsm_train.py --config configs/fgsm_l1unstruct.yml  #for IMP training on CIFAR10
python fgsm_train.py --config configs/fgsm_l1unstruct_nrs.yml  #for IMP+NRSLoss training on CIFAR10
#other training instances can be found in "configs" folder
```
 

##### Auto-LiRPA training
```
python lirpa_train.py --config configs/cibp_l1unstruct.yml  #for IMP training on CIFAR10
python lirpa_train.py --config configs/cibp_l1unstruct_nrs.yml  #for IMP+NRSLoss training on CIFAR10
#other training instances can be found in "configs" folder
```

### Note:
The auto-LiRPA lib is a slight modification from the original version: https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/auto_LiRPA to enable auto-LiRPA training with various iterative pruning methods.
