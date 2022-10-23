# Introduction

Thank you for reviewing carefully our work.



This code is based on this [project](https://github.com/Impression2805/CVPR21_PASS).



This code includes our main comparative experiments with [LwF](https://arxiv.org/pdf/1606.09282.pdf) and [PASS](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf). The dataset are [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet-Subset](https://drive.google.com/file/d/1gRoYB0_bemvImFE3I30lmFsawOidQJP3/view?usp=sharing).



 ## Run on CIFAR-100

LwF

```
python main.py --data_name cifar100 --ref L
```

LwF.ProtoAug

```
python main.py --data_name cifar100 --ref LPA
```

LwF+PRE

```
python main.py --data_name cifar100 --ref LPAME
```

PASS

```
python main.py --data_name cifar100 --ref LPAS
```

PASS+PRE

```
python main.py --data_name cifar100 --ref LPASME
```





## Run on ImageNet-Subset

Before testing on ImageNet-Subset,  please download the [relevant dataset](https://drive.google.com/file/d/1gRoYB0_bemvImFE3I30lmFsawOidQJP3/view?usp=sharing), decompress it and put it in. './dataset'.



LwF

```
python main.py --data_name TinyImageNet --ref L
```

LwF.ProtoAug

```
python main.py --data_name TinyImageNet --ref LPA
```

LwF+PRE

```
python main.py --data_name TinyImageNet --ref LPAME
```

PASS

```
python main.py --data_name TinyImageNet --ref LPAS
```

PASS+PRE

```
python main.py --data_name TinyImageNet --ref LPASME
```

