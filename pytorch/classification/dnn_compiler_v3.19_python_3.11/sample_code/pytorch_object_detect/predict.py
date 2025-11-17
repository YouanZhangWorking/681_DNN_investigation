'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import os
import torch
from model.generic_ssdlite import create_generic_ssdlite
from model.config import generic_ssd_config
from utils.box_utils import convert_locations_to_boxes
from transforms.transforms import *
import cv2
import matplotlib.pyplot as plt
from utils.quantization import load_quantized_model
from utils.misc import set_random_seed

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

def draw_result(image,box):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_h,image_w, c = image.shape
    yc,xc,h,w = box
    x_ul = xc-0.5*w
    y_ul = yc-0.5*h
    x_ul*=image_w
    y_ul*=image_h
    w*=image_w
    h*=image_h
    image = cv2.rectangle(image,(int(x_ul),int(y_ul)),(int(x_ul+w),int(y_ul+h)),(0,255,0),1)
    plt.imshow(image)
    plt.show()

def find_max_box(bbox,conf,label_map):
    best_box = None
    best_conf = -1
    class_pred = None
    for b, c in zip(bbox[0],conf[0]):
        b = b.cpu().detach().numpy()
        c = c.cpu().detach().numpy()
        if c[1] > best_conf or c[2] > best_conf:
            if c[2]>c[1]:
                class_pred=label_map[2]
            else:
                class_pred=label_map[1]
            best_conf = max(c[1],c[2])
            best_box = b
    return best_box, best_conf,class_pred

# Define hook to capture output
outputs = {}
def hook_fn(module, input, output):
    outputs[module] = output

if __name__ == "__main__":
    set_random_seed(0)  # Set seed before creating random data

    # Set config import here:
    from model.config.generic_ssd_config import image_size,priors,center_variance,size_variance,image_mean,image_std

    device = torch.device("cpu")
    model_file = './saved_models/results/Epoch-149-Quantized_True.pth'
    label_file = './model/config/label_map/circle-rectangle-model-labels.txt'
    image_dir = './dummy_dataset/validation/images'
    quantize = True

    # load state_dict
    state_dict = torch.load(model_file,map_location=device)
    
    # Get info from model.config
    size = image_size
    priors = priors.to(device)

    # object classes
    class_names = tuple([name.strip() for name in open(label_file).readlines()])
    
    label_map = {}
    for i, cn in enumerate(class_names):
        label_map[i]= cn
    transform = ImageTransform(size,mean=image_mean, std=image_std)

    config = generic_ssd_config
    num_classes = len(class_names)

    model = create_generic_ssdlite(num_classes, quantize=quantize)
    model.to(device)    

    if quantize:
        model = load_quantized_model(model,config.fuse_layer_list,model_file,device=device)
    else:
        state_dict = torch.load(model_file,map_location=device)
        model.load_state_dict(state_dict)
        
    model.eval()
    image_files = os.listdir(image_dir)

    # Loop through and predict images
    for f in image_files:
        image = cv2.imread(os.path.join(image_dir,f),-1)
        image_orig = np.copy(image)

        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        conf, bbox = model.forward(image)

        bbox = convert_locations_to_boxes(bbox, priors, center_variance, size_variance)
        
        box, max_c, class_name = find_max_box(bbox,conf, label_map)
        print(f"Class pred: {class_name}, conf:{max_c}")
        draw_result(image_orig, box)


