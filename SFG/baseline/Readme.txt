###################Training
To train baseline, run "python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 512". 

###################Validation
To validate baseline, run "python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 512 --valid". 

  
For more details, please see our paper.
  @inproceedings{ge2022SFG,
  author = {Lin Ge, Xingyue Wei, Yayu Hao, Jianwen Luo, Yan Xu},
  title = {Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network},
  booktitle = {},
  year = {2022}
  }
