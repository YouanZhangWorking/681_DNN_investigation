***Copyright 2025 Sony Semiconductor Solutions Corporation. This is the UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, sold, and distributed in any form or by any means without prior explicit permission in writing of Sony Semiconductor Solutions Corporation.***

This project contains code to train a simple object detection DNN in PyTorch which is compatible with the Sony IMX681 DNN compiler. This is a simple reference training pipeline with a synthetic training dataset containing white circles and rectangles randomly placed on a dark background. The purpose is to verify imx681 compiler functionality with a simple reference model. The model architecture and training parameters aren't fully optimized. 

## Environment
Python 3.8.10  
torch 2.4.1  
torchvision 0.19.1  
opencv-python 4.11  

## Dataset

Generate the synthetic training and validation datasets by running the script below:

```python dummy_dataset/prepare_dummy_dataset.py```

This script will generate 10k training samples and 2k validation samples in the "dummy_dataset" folder. The directory structure should match below: 

dummy_dataset/  
├── train/  
│   ├── images/  
│   └── labels/  
└── validation/  
|    ├── images/  
|    └── labels/  

## Training

Set the training parameters in config/train_config.yaml and run the training script:

```python train.py config/train_config.yaml```

***It is recommended to use the default training parameters provided in train_config.yaml***

## Testing
To perform a sanity check on the model before passing to the DNN compiler, we visualize predictions on a test dataset. Change the configuration parameters at the top of predict.py and run: 

```python predict.py```

This will run inference on each image in the specified directory and draw the highest confidence bounding box on the image. It will also print the predicted class and confidence. If working properly, the predicted boxes should visually match the objects location and class (circle or rectangle). 

**TODO: Add mAP calculation**
## Generate anchor box file for IMX681
**Important: This section assumes this code is in dnn_parser-python-3-10/sample_code/**

To generate the SSD prior boxes in the format the compiler expects, change the output file location at the top of the generate_anchors.py script and run:

```python generate_anchors.py```

## Compile for IMX681
**Important: This section assumes this code is in dnn_parser-python-3-10/sample_code/**

change the configuration parameters in the compile_for_imx681.py script and run:

```python compile_dnn_pytorch.py```

