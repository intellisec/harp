# Holistic Adversarially Robust Pruning

Neural networks can be drastically shrunk in size by removing redundant parameters. While crucial for the deployment 
on resource-constraint hardware, oftentimes, compression comes with a severe drop in accuracy and lack of adversarial 
robustness. Despite recent advances, counteracting both aspects has only succeeded for moderate compression rates so 
far. We propose a novel method, HARP, that copes with aggressive pruning significantly better than prior work. For this, 
we consider the network holistically. We learn a global compression strategy that optimizes how many parameters 
(compression rate) and which parameters (scoring connections) to prune specific to each layer individually. Our method 
fine-tunes an existing model with dynamic regularization, that follows a step-wise incremental function balancing the 
different objectives. It starts by favoring robustness before shifting focus on reaching the target compression rate 
and only then handles the objectives equally. The learned compression strategies allow us to maintain the pre-trained 
model’s natural accuracy and its adversarial robustness for a reduction by 99% of the network’s original size. 
Moreover, we observe a crucial influence of non-uniform compression across layers.

For further details please consult the [conference publication](https://intellisec.de/pubs/2023-iclr.pdf).

The figure below shows an overview of pruning weights of a VGG16 model for CIFAR-10 (left) and SVHN (right) with PGD-10 
adversarial training. Solid lines show the natural accuracy. Dashed lines represent the robustness against AUTOATTACK.

<img src="https://intellisec.de/research/harp/overview.svg" width="1000" /><br />


## Publication
A detailed description of our work will be presented at [ICLR](https://iclr.cc/Conferences/2023) in May 2023. 
If you would like to cite our work, please use the reference as provided below:

```
@InProceedings{Zhao2023Holistic,
author    = {Qi Zhao and Christian Wressnegger},
booktitle = {Proc. of the International Conference on Learning Representations (ICLR)},
title     = {Holistic Adversarially Robust Pruning},
year      = {2023},
month     = may,
}
```

A preprint of the paper is available [here](https://intellisec.de/pubs/2023-iclr.pdf).

## Code

### Prerequisites
Our work is accomplished by 
[![Generic badge](https://img.shields.io/badge/Library-Pytorch-green.svg)](https://pytorch.org/) with CUDA 11.1.
Following commands help to obtained the needed environment via `conda` and `pip`. Other Pytorch versions are found
[here](https://pytorch.org/get-started/previous-versions/).
```
conda create -n harp python=3.8
conda activate harp
pip install numpy pyyaml scikit-learn
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchattacks torchvision
```

### Adversarial pre-training
We use 100 epochs in adversarial pre-training for small-scale datasets. Run following to do 
PGD-AT([Madry et al.](https://arxiv.org/pdf/1706.06083.pdf)) on VGG16 model on CIFAR-10.

```
python train.py --arch 'vgg16_bn' --dataset 'CIFAR10' --exp-mode 'pretrain' --adv_loss 'pgd' --epochs 100
```

- Download pre-trained models:

  Here we additionally offer the robust pre-trained models that are used in our experiments. Run script `download_pretrain.sh` to download all pre-trained 
models.

### Pruning strategy training
In pruning stage, we have three different modes i.e. `score-prune`, `rate-prune` and `harp-prune`, which refers to 
training mask scores only, layer pruning rates only and both concurrently. For small-scale datasets, we use 20 epochs 
in the pruning stage. To prune e.g. a VGG16 with PGD-AT by HARP, we need to set the pruning granularity `prune_reg`, 
the target compression rate `k`, the incremental regularization factor for hw-loss `gamma` and the initial strategy 
`stg_id`. 

```
python train.py --arch 'vgg16_bn' --dataset 'CIFAR10' --exp-mode 'harp_prune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni' --adv_loss 'pgd' --gamma 0.01 --epochs 20
```

### Fine-tuning model weights

After HARP's pruning, we change to fine-tuning mode by selecting correspondingly from `score-finetune`, `rata-finetune` 
and `harp-finetune`. In addition, the same adversarial loss `adv_loss`, target rate `k`, pruning 
granularity `prune_reg` and initial strategy ID `stg_id` (here only for finding directory) require to be defined as 
in the pruning stage. 

```
python train.py --arch 'vgg16_bn' --dataset 'CIFAR10' --exp-mode 'harp_finetune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni' --adv_loss 'pgd' --epochs 100
```

### Evaluate pruned model
Script `test.py` helps to evaluate on the robustness against different adversarial attacks for fine-tuned sparse models. 
Run following to implement the evaluation on VGG16 for CIFAR10.

```
python test.py --arch 'vgg16_bn' --dataset 'CIFAR10' --exp-mode 'harp_finetune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni'
``` 


### Experiment with ResNet50 on ImageNet

Experiments of ImageNet are conducted on ResNet50 with the same setting as in Free-AT([Shafahi et al.](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf)). 
Following commands help to accomplish a HARP weight pruning on ImageNet. Note that data normalization is required to 
match the setting of the inherited pre-trained model directly from Free-AT.

* HARP Pruning
```
python train.py --arch 'ResNet50' --dataset 'imagenet' --normalize --exp-mode 'harp_prune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni' --trainer 'freeadv' --gamma 0.1 --epochs 5
```
* HARP Fine-tuning
```
python train.py --arch 'ResNet50' --dataset 'imagenet' --normalize --exp-mode 'harp_finetune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni' --trainer 'freeadv' --epochs 25
```
* Test ImageNet model
```
python test.py --arch 'ResNet50' --dataset 'imagenet' --normalize --exp-mode 'harp_finetune' --k 0.01 --prune_reg 'weight' --stg_id '010_uni' --trainer 'freeadv'
```
 

---
### Implementation of non-uniform related work
In addition, repositories `hydra-nonuniform` and `radmm-nonuniform` provide the code to run [Hydra](https://proceedings.neurips.cc/paper/2020/file/e3a72c791a69f87b05ea7742e04430ed-Paper.pdf)
and [Robust-ADMM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf)
with non-uniform strategies.
