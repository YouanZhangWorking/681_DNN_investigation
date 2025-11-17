'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import sys
import torch
from model.generic_ssdlite import create_generic_ssdlite
from model.config import generic_ssd_config
from utils.quantization import load_quantized_model
from predict import ImageTransform
from utils.misc import set_random_seed


# Windows file paths for this part
sys.path.append(r"../../")

import dnn_compiler

if __name__ == "__main__":
    set_random_seed(0)  # Set seed before creating random data

    # Set config import here:
    from model.config.generic_ssd_config import image_size,image_mean,image_std
    transform = ImageTransform(image_size,mean=image_mean, std=image_std)

    model_path = r'.\saved_models\results\Epoch-149-Quantized_True.pth'
    label_file = r'.\model\config\label_map\circle-rectangle-model-labels.txt'
    compiler_config = r'..\..\configs\imx681_pytorch_detection_i2c.cfg'
    
    overrides=[]
    overrides.append("OUTPUT_ENDIANNESS=LITTLE") # LITTLE endianness for simulator, BIG for sensor

    # Device to load model on to
    device = torch.device('cpu')

    # load state_dict
    state_dict = torch.load(model_path,map_location=device)

    # object classes
    class_names = tuple([name.strip() for name in open(label_file).readlines()])

    # set up
    config = generic_ssd_config
    num_classes = len(class_names)
    model_cpu = create_generic_ssdlite(num_classes, quantize=True)
    model_cpu.to(device)

    # Load the quantized model
    model_cpu = load_quantized_model(model_cpu,config.fuse_layer_list,model_path,device=device)
    
    dnn_compiler.run(compiler_config, model_cpu, config_overrides=overrides)


