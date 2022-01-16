# Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network

By Lin Ge, Xingyue Wei, Yayu Hao, Jianwen Luo, Yan Xu.


## Introduction
This is the code for SFG. For more details, please refer to our paper.

Code has been tested with Python 3.6 and MXNet 1.5.

## Datasets

We use the dataset after affine registration.

## Training

- to train D_SFG, go to D_SFG direction and  run `python train_D_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.

- to train S_SFG, go to S_SFG direction and run `python train_S_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.

## Reference

```
@inproceedings{zhao2020maskflownet,
  author = {Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I-Chao and Xu, Yan},
  title = {MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```


