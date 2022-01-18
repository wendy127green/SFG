# Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network

By Lin Ge, Xingyue Wei, Yayu Hao, Jianwen Luo, Yan Xu.


## Introduction
Registration of multiple stained images is
a fundamental task in histological image analysis. In
supervised methods, obtaining ground-truth data with
known correspondences can be very laborious and timeconsuming.
Thus, unsupervised methods are expected.
However, unsupervised learning methods ease the burden
of manual annotation, but often at the cost of inferior
results. In addition, the registration of histological images
suffers from appearance variance due to multiple staining,
repetitive texture and section missing during making tissue
sections. To deal with these challenges, we propose an
unsupervised structural feature guided convolutional neural
network (SFG). Structural features are not only the key
information for pathological diagnosis, but they are also robust
to multiple staining. SFG consists of two components,
i.e., dense structural component and sparse structural
component. Dense structural component uses structural
feature maps of the whole images as structural consistency
constraints, which is comprehensive and represents
local contextual information. Sparse structural component
utilizes the distance of matched key points as structural
consistency constraints, because that matching key points
in a pair of images emphasizes the matching of the significant
structures, which imply global information. The
combination of dense and sparse structural component
can overcome repetitive texture and section missing. The
proposed method was evaluated on a public histological
dataset (ANHIR) and is ranked 1st.

This is the code for SFG. For more details, please refer to our paper.

Code has been tested with Python 3.6 and MXNet 1.5.

## Datasets

We use the dataset after affine registration.

## Training

- To train unsupervised registration, go to D_SFG direction and run `python train_D_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 0 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`. Or go to S_SFG direction and run `python train_S_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 0 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`

- To train supervised registration with landmarks, go to S_SFG direction and run `python train_S_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.
The input data of key points should be landmarks.

- To train D_SFG (Dense SFG), go to D_SFG direction and  run `python train_D_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 100 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.

- To train S_SFG (Sparse SFG), go to S_SFG direction and run `python train_S_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.
The input data of key points should be auto-obtained key points.

- To train C_SFG/SFG (dense and sparse SFG), go to D_SFG direction and run `python train_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 1024_with_lung_lesion_and_6points_as_eva_`.
## Reference

```
@inproceedings{zhao2020maskflownet,
  author = {Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I-Chao and Xu, Yan},
  title = {MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```


