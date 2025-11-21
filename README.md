# Perceptual models GitHub repository.

This library hosts the code to instance the neural networks that are supported in MONAI's percpetual loss.

The weights for the models are stored in Huggingface here: https://huggingface.co/MONAI/checkpoints

## MedicalNet models

With MONAI's standard perceptual loss, you can use the 3D ResNet10 and ResNet50 MedicalNet models, originally released in https://github.com/Tencent/MedicalNet. These resnets are different backbones of the MedicalNet training regime [1]. The network was trained on a collection of segmentation 
datasets for medical imaging, and used in numerous transfer learning tasks.

## RadImageNet models

The RadImageNet is a large multi-modal medical imaging dataset: https://www.radimagenet.com, and also a repository containing 2D networks trained on the
database: https://github.com/BMEII-AI/RadImageNet, and providing pre-trained models. 
In MONAI, we support ResNet50. 

`[1] Chen, Sihong and Ma, Kai and Zheng, Yefeng, Med3D: Transfer Learning for 3D Medical Image Analysis (2019), arXiv:1904.00625.`
`[2] Mei, Xueyan and Liu, Zelong and Robson, Philip M. and Marinelli, Brett and Huang, Mingqian and Doshi, Amish and Jacobi, Adam and Cao, Chendi and Link, Katherine E. and Yang, Thomas and Wang, Ying and Greenspan, Hayit and Deyer, Timothy and Fayad, Zahi A. and Yang, Yang; RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning (2022).` 