'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import cv2
import numpy as np
import random 
import os
from tqdm import tqdm

def random_gaussian_blur(image):
    """
    Randomly apply Gaussian blur to an image with kernel size of 1, 3, 5, or 7.
    Kernel size of 1 means no blur is applied.
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        numpy array: Blurred image (or original if kernel_size=1)
    """
    # Possible kernel sizes (1 means no blur)
    kernel_sizes = [1, 3, 5, 7]
    
    # Randomly choose kernel size
    kernel_size = random.choice(kernel_sizes)
    
    # If kernel_size is 1, return original image
    if kernel_size == 1:
        return image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to an image.
    
    Args:
        image (numpy.ndarray): Input image array
        mean (float): Mean of the Gaussian noise (default: 0)
        std (float): Standard deviation of the Gaussian noise (default: 25)
    
    Returns:
        numpy.ndarray: Image with added Gaussian noise, clipped to original image type range
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, std, image.shape)
    
    # Add noise to image
    noisy_image = image + noise
    
    # Clip the values to be in valid range
    if image.dtype == np.uint8:
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    else:
        # For floating point images, clip to [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

def create_dataset_structure(base_path='dataset'):
    """
    Creates a dataset directory structure with 'images' and 'labels' subdirectories.
    
    Args:
        base_path (str): Base path for the dataset directory (default: 'dataset')
    """
    # Create directories
    dirs_to_create = [
        base_path,
        os.path.join(base_path, 'images'),
        os.path.join(base_path, 'labels')
    ]
    
    # Create each directory if it doesn't exist
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")

def generate_frame(size=(120,160),r_min=10,r_max=20,h_min=20,h_max=40, w_min=20, w_max=40,pad=1.0,border=20):
    bg = random.randint(0,25)
    image = np.zeros(size)+bg
    intensity = random.randint(bg+175,255)
    if random.random() <= 0.5:
        r = random.randint(r_min,r_max)
        x = random.randint(r+border,int(size[1]-pad*r*2)-border) # r+20, (160-1.0*r*2)-20; center-X
        y = random.randint(r+border,int(size[0]-pad*r*2)-border)  # center-Y
        image = cv2.circle(image,(x,y),r,(intensity),-1)  # cv2.circle(img, center, radius, color, thickness), -1表示填充圆形
        bbox = [0,y/size[0],x/size[1],(2*r)/size[0],(2*r)/size[1]] # box[cls, y, x, h, w]
    else:
        w = random.randint(w_min,w_max)
        h = random.randint(h_min,h_max)
        x = random.randint(border,int(size[1]-pad*w)-border)
        y = random.randint(border,int(size[0]-pad*h)-border)
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(intensity),-1)
        xc, yc = (x + 0.5*w), (y + 0.5*h)
        bbox = [1,yc/size[0],xc/size[1],h/size[0],w/size[1]] # box[cls, y, x, h, w]
    # image = random_gaussian_blur(image)
    image = image.astype(np.uint8)
    # image = add_gaussian_noise(image,mean=((bg + intensity)/2.),std=5)
    return image, bbox

if __name__ == "__main__":
    data_path = 'dummy_dataset'
    train_path = os.path.join(data_path,'train')
    val_path = os.path.join(data_path,'validation')
    train_samples = 600
    val_samples = 173
    
    # Create the subfolders to write the generated images to:
    create_dataset_structure(train_path)
    create_dataset_structure(val_path)

    for i in tqdm(range(train_samples)):
        image, bbox = generate_frame()
        cv2.imwrite(os.path.join(train_path,f'images/{i}.jpg'),image)
        with open(os.path.join(train_path,f'labels/{i}.txt'),'w') as fi:
            fi.write(' '.join([str(x) for x in bbox]))

    print(f"Finished generating {train_samples} images (train)")

    for i in tqdm(range(val_samples)):
        image, bbox = generate_frame()
        cv2.imwrite(os.path.join(val_path,f'images/{i}.jpg'),image)
        with open(os.path.join(val_path,f'labels/{i}.txt'),'w') as fi:
            fi.write(' '.join([str(x) for x in bbox]))
    
    print(f"Finished generating {val_samples} images (validation)")
