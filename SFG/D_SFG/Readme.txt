###################Training
To train D_SFG (Sparse SFG), run "python train_D_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 512". 

###################Validation
To validate D_SFG (Sparse SFG), run "python train_D_SFG.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 512 --valid". 


##################Dataset

We have provided our datasets with image size of 512*512 pixels in /datasets dictory. 
(1)The /datasets/512after_affine includes the training and test images as well as the key points.
  For trainging images, the key points are consistent with the data provided by ANHIR. 
  For test images, we invited an expert to manually annotate 6 pairs of landmarks on each evaluation image pair for analyzing our methods. If you need to use these data, please cite our paper as follows:
  @inproceedings{ge2022SFG,
  author = {Lin Ge, Xingyue Wei, Yayu Hao, Jianwen Luo, Yan Xu},
  title = {Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network},
  booktitle = {},
  year = {2022}
  }

(2)The "/home/gelin/SFG/D_SFG/datasets/multi_siftflowmap_512/" includes the multi-scale SIFT information for training images used in training process. 
  The "/home/gelin/SFG/D_SFG/datasets/siftflowmap_512/" includes the single-scale SIFT information for training images used in training process.
  Please go to /SFG/SIFTflow_map_obtain/ to see the code to obtain the key points.
  
  
  
For more details, please see our paper.
  @inproceedings{ge2022SFG,
  author = {Lin Ge, Xingyue Wei, Yayu Hao, Jianwen Luo, Yan Xu},
  title = {Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network},
  booktitle = {},
  year = {2022}
  }