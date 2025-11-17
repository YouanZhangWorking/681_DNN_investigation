'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path
import numpy as np
from transforms.transforms import *

class ImageTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image

class CalibrationDataset(Dataset):
    """Dataset for calibration images"""
    def __init__(self, image_dir, transform, input_size=(120, 160)):
        """
        Args:
            image_dir (str): Directory with images
            input_size (tuple): (height, width) for resizing
        """
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.transform = transform
        # Get all image files
        # self.image_files = []
        self.image_files = list(self.image_dir.glob('*.jpg'))
        # for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        #     self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
        #     self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read image
        img_path = str(self.image_files[idx])
        image = cv2.imread(img_path,-1)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = self.transform(image)
        # image = image.unsqueeze(0)
        
        # Dummy target (0) since we don't need it for calibration
        target = 0
        
        return image, target

def create_calibration_loader(image_dir, transform, input_size=(120, 160), batch_size=1):
    """
    Create a DataLoader for calibration images.
    
    Args:
        image_dir (str): Directory containing calibration images
        input_size (tuple): (height, width) for input images
        batch_size (int): Batch size for loader
        
    Returns:
        DataLoader for calibration
    """
    try:
        dataset = CalibrationDataset(image_dir, transform, input_size)
        loader = DataLoader(dataset, 
                          batch_size=batch_size,
                          shuffle=False)
        
        print(f"Created calibration loader with {len(dataset)} images")
        return loader
        
    except Exception as e:
        print(f"Error creating calibration loader: {str(e)}")
        return None