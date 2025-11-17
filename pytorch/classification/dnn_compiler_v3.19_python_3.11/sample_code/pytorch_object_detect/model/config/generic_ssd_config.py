'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import numpy as np
from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = (120, 160)
image_mean = np.array([127])  
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec((8, 10), 15, SSDBoxSizes(10, 20), [1,2,3]),
    SSDSpec((4, 5), 30, SSDBoxSizes(20, 30), [1,2,3]),
    SSDSpec((1, 1), 120, SSDBoxSizes(30, 40), [1,2,3])
]

priors = generate_ssd_priors(specs, image_size, reduce_boxes_in_lowest_layer=False)

fuse_layer_list = [   # list of layers to fuse together (Model specific)
        ['base_net.in_conv.conv', 'base_net.in_conv.bn', 'base_net.in_conv.relu'],
        *[[f'base_net.conv{i}.conv1', 
            f'base_net.conv{i}.bn1', 
            f'base_net.conv{i}.relu1'] for i in range(1,13)],
        *[[f'base_net.conv{i}.conv2', 
            f'base_net.conv{i}.bn2', 
            f'base_net.conv{i}.relu2'] for i in range(1,13)],
        ['base_net.conv13.conv', 'base_net.conv13.bn', 'base_net.conv13.relu'],
        *[
            *[[f'classification_headers.{i}.conv1',
            f'classification_headers.{i}.bn1',
            f'classification_headers.{i}.relu1'] for i in range(0,3)],
            *[[f'classification_headers.{i}.conv2',
            f'classification_headers.{i}.bn2'] for i in range(0,3)],
            *[[f'regression_headers.{i}.conv1',
            f'regression_headers.{i}.bn1',
            f'regression_headers.{i}.relu1'] for i in range(0,3)],
            *[[f'regression_headers.{i}.conv2',
            f'regression_headers.{i}.bn2'] for i in range(0,3)]
        ]
    ]

