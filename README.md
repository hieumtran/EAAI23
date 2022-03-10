# EAAI23 - Matching Songs to Emotive Faces
## Abstract
An artificial intelligence (AI) system is “human-aware” when humans play an interactive role in the system. In this student-faculty collaborative research, we propose the development of a human-aware AI system to detect the user’s emotion and make corresponding song playback suggestions. Various convolutional neural network (CNN) architectures will be applied for emotion classification from real-time facial image captures. The system will also analyze songs in the user’s Spotify history, applying deep learning to categorize songs based on lyrics and audio, combined with attributes already assigned by Spotify. The system will then recommend songs matching the user’s mood. A paper on this work will be submitted to the Educational Advances in Artificial Intelligence (EAAI) symposium for peer review, with accepted papers published in the international Association for the Advancement of Artificial Intelligence (AAAI) 2023 conference proceedings.

## Requirements
* Python >= 3.8 (3.9 recommended)
* Torch >= 1.10 (1.10.2=+cu113 recommended)
* Pandas >= 1.2 (1.2.4 recommended)
* Tensorflow = 2.4.0
* Numpy >= 1.19 (1.19.3 recommended)

## Folder structure
```
EAAI23/
│
├── AlexNet_pytorch/ - Implementation of AlexNet in PyTorch
│
├── AlexNet_tensorflow/ - Implementation of AlexNet in Tensorflow
│
├── data/ - AffectNet in .npy form (image-path and annotation)
│
├── PyTorch_reg/ - PyTorch implementation for regression
|   ├── loss_function.py - Eucledian L2 distance
│   └── design/ - Architecture implementation
|         ├── resnet.py - ResNet (BottleNeck), Spatial Transformer
|         ├── vgg_simple.py  - VGG-16
|         └── sp_trans.py - Spatial Transformer
|
...
```

## Authors
* [Hieu Tran](https://github.com/hieumtran)
* [Anh Do](https://github.com/anhphuongdo34) 

<!-- └──, ├──, │  --> 