# NAS-HBO
## Introduction
**A Novel High-precision Bilevel Optimization Method for 3D Pulmonary Nodule Classification** 

Sorry, the paper is currently under review. Once the paper is published, we will release the code.
## Results

### NAS-HBO

| model               | Accu.       | Sens. | Spec. | F1 Score | para.(M)    |
| ------------------- | ----------- | ----- | ----- | -------- | ----------- |
| Multi-crop CNN      | 87.14       | -     | -     | -        | -           |
| Nodule-level 2D CNN | 87.30       | 88.50 | 86.00 | 87.23    | -           |
| Vanilla 3D CNN      | 87.40       | 89.40 | 85.20 | 87.25    | -           |
| ADNN                | 90.11       | -     | -     | -        | -           |
| DeepLung            | 90.44       | 81.42 | -     | -        | 141.57      |
| AE-DPN              | 90.24       | 92.04 | 88.94 | 90.45    | 678.69      |
| NASLung             | 90.77       | 85.37 | 95.04 | 89.29    | 16.84       |
| NAS-qa              | 90.85       | 86.04 | 91.02 | 88.89    | -           |
| **NAS-HBO(ours)** | **91.51(top)** | 90.32(second) | 91.82(second) | 89.46(second) | **12.79(top)** |

### Verification results

Due to limited experimental conditions, the same hyperparameters were used and we did not conduct particularly fine tuning. Therefore, the experimental results may be further improved.

| Model  | Accu.  | Sens.  | Spec.  | F1 Score | para. |
| ------ | ------ | ------ | ------ | -------- | ----- |
| Fold-5 | 92.391 | 94.595 | 90.909 | 90.909   | 12.79 |
| Fold-6 | 91.304 | 85.714 | 94.521 | 87.805   | 12.79 |
| Fold-7 | 91.111 | 85.714 | 93.548 | 85.714   | 12.79 |
| Fold-8 | 89.423 | 91.837 | 87.272 | 89.109   | 12.79 |
| Fold-9 | 93.333 | 93.750 | 92.857 | 93.750   | 12.79 |

## Usage
To run our code, you only need one GeForce RTX 4090(24G memory).

#### Preprocessing
You need to download the LUNA16 dataset by yourself and adjust the corresponding paths.
```
prepare.py \\
```
```
nodclsgbt.py \\
```
#### Search on the LUNA16 dataset

The number of pre-training epochs W in the program is the number of epochs used to solely train W. The epoch when the validation set first reaches its optimal value needs to be determined again if you plan to use your own dataset, as detailed in section 4.1 of the paper.

```
train_search.py \\
```
#### Evaluation on the LUNA16 dataset

Some randomness is introduced in certain aspects of the code, such as partial channel connections, to allow for fluctuations in results. Due to the 37-hour training time required for the model, we did not finely tune the hyperparameters, and thus the experimental results we present may not represent the optimal values.

```
python train_3D_continuation.py \\
```

### Requirements

To ensure the code can run, we provide versions of some libraries.

- python-3.7.13
- numpy-1.21.5
- pytorch-1.21.1
- pandas-1.3.5
- opencv-python-4.8.1

## Acknowledgement 

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

 **This code is based on the implementation of  [DARTS](https://github.com/quark0/darts)，[Fair-DARTS](https://github.com/xiaomi-automl/FairDARTS)，[PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)，[ProxylessNAS](https://github.com/MIT-HAN-LAB/ProxylessNAS)，[NAS-Lung](https://github.com/fei-hdu/NAS-Lung)，[DeepLung](https://github.com/uci-cbcl/DeepLung), [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet), [LayerCAM](https://github.com/PengtaoJiang/LayerCAM-jittor) and [Torch-cam](https://github.com/frgfm/torch-cam).**

## Selected References

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

- W. Zhu, C. Liu, W. Fan, X. Xie, Deeplung: Deep 3d dual path nets for automated pulmonary nodule detection and classification,  2018 IEEE winter conference on applications of computer vision (WACV), IEEE2018, pp. 673-681.
- H. Liu, K. Simonyan, Y. Yang, Darts: Differentiable architecture search, arXiv preprint arXiv:1806.09055, (2018).
- Y. Xu, L. Xie, X. Zhang, X. Chen, G.-J. Qi, Q. Tian, H. Xiong, Pc-darts: Partial channel connections for memory-efficient architecture search, arXiv preprint arXiv:1907.05737, (2019).
- H. Jiang, F. Shen, F. Gao, W. Han, Learning efficient, explainable and discriminative representations for pulmonary nodules classification, Pattern Recognition, 113 (2021) 107825.
- X. Chu, T. Zhou, B. Zhang, J. Li, Fair darts: Eliminating unfair advantages in differentiable architecture search,  European conference on computer vision, Springer2020, pp. 465-480.
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks,  Proceedings of the IEEE conference on computer vision and pattern recognition2018, pp. 4510-4520.

- S.G. Armato III, G. McLennan, L. Bidaut, M.F. McNitt‐Gray, C.R. Meyer, A.P. Reeves, B. Zhao, D.R. Aberle, C.I. Henschke, E.A. Hoffman, The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans, Medical physics, 38 (2011) 915-931.

- K. Kuan, M. Ravaut, G. Manek, H. Chen, J. Lin, B. Nazir, C. Chen, T.C. Howe, Z. Zeng, V. Chandrasekhar, Deep learning for lung cancer detection: tackling the kaggle data science bowl 2017 challenge, arXiv preprint arXiv:1705.09435, (2017).

- P.-T. Jiang, C.-B. Zhang, Q. Hou, M.-M. Cheng, Y. Wei, Layercam: Exploring hierarchical class activation maps for localization, IEEE Transactions on Image Processing, 30 (2021) 5875-5888.

- Cai, H., Zhu, L., & Han, S. (2018). Proxylessnas: Direct neural architecture search on target task and hardware. arXiv preprint arXiv:1812.00332.
