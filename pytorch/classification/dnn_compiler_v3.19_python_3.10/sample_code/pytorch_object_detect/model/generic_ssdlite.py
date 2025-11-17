'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import torch
import math
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from collections import OrderedDict
from .ssd import SSD, GraphPath
from .config import generic_ssd_config as config
import torch.nn.quantized as nnq

ACTIVATION=nn.ReLU(inplace=False)

# ACTIVATION = torch.ao.nn.quantized.ReLU6() # Doesn't work with CUDA backend.
# ACTIVATION = torch.nn.ReLU6() # Works with CUDA backend, but fuse operation not supported.


# Conv2D -> BN (optional) -> ReLU
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', ACTIVATION)
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            ('relu', ACTIVATION)
        ]))
    
# DepthwiseConv2D -> BN (optional) -> ReLU
def dw_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', ACTIVATION)
        ]))
    else:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)),
            ('relu', ACTIVATION)
        ]))

def ConvBlock1(
    in_channels, 
    out_channels, 
    channel_multiplier=1, 
    kernel_size=1, 
    stride=1, 
    padding=0):
    
    """
    Depthwise Conv2D -> BN -> ReLU -> Pointwise Conv2D -> BN -> ReLU
    """
    return Sequential(OrderedDict([
        ('conv1', Conv2d(in_channels=in_channels, out_channels=in_channels*channel_multiplier, 
                         kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=False)),
        ('bn1', BatchNorm2d(in_channels*channel_multiplier)),
        ('relu1', ACTIVATION),
        ('conv2', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)),
        ('bn2', BatchNorm2d(out_channels)),
        ('relu2', ACTIVATION)
    ]))

def PointWiseHeader(
    in_channels, 
    out_channels, 
    channel_multiplier=1, 
    stride=1):
    
    """
    Pointwise Conv2D -> BN -> ReLU -> Pointwise Conv2D -> BN
    """
    return Sequential(OrderedDict([
        ('conv1', Conv2d(in_channels=in_channels, out_channels=in_channels*channel_multiplier, 
                         kernel_size=1, stride=stride, padding=0, bias=False)),
        ('bn1', BatchNorm2d(in_channels*channel_multiplier)),
        ('relu1', ACTIVATION),
        ('conv2', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)),
        ('bn2', BatchNorm2d(out_channels))
    ]))

def SeperableConv2d(
    in_channels, 
    out_channels, 
    channel_multiplier=1, 
    kernel_size=1, 
    stride=1, 
    padding=0):
    
    """
    Depthwise Separable Conv2D
    """
    return Sequential(OrderedDict([
        ('conv1', Conv2d(in_channels=in_channels, out_channels=in_channels*channel_multiplier, 
                         kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, bias=False)),
        ('bn1', BatchNorm2d(in_channels*channel_multiplier)),
        ('relu1', ACTIVATION),
        ('conv2', Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)),
        ('bn2', BatchNorm2d(out_channels))
    ]))
    
def create_generic_ssdlite(
    num_classes, 
    quantize=True, 
    is_test=False,
    reg_in_ch=64):
    """
    Construct model architecture.
    """
    # Feature Extractor
    base_net = Backbone().features
    
    # Feature map layers to be fed to classification and regression headers
    source_layer_indexes = [
        9, 11, 14
    ]
    
    extras = ModuleList([
        # no extra modules
    ])

    # Box regression headers
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(reg_in_ch), out_channels=32, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(reg_in_ch), out_channels=32, kernel_size=3, padding=1),
        PointWiseHeader(in_channels=round(reg_in_ch), out_channels=32)
    ])

    # Classification headers
    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(reg_in_ch), out_channels=8 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=round(reg_in_ch), out_channels=8 * num_classes, kernel_size=3, padding=1),
        PointWiseHeader(in_channels=round(reg_in_ch), out_channels=8 * num_classes)
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, 
               is_test=is_test, config=config, quantize=quantize)

    
"""
Feature Extractor Backbone
"""
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        input_channel = 1
        first_channel = 8

        # list of layers
        self.features = [conv_bn_relu(input_channel, first_channel, kernel_size=3, stride=1, padding=1), # 160 x 120
                         ConvBlock1(8,8,kernel_size=3,stride=2,padding=1), # 80 x 60
                         ConvBlock1(8,8,kernel_size=3,stride=1,padding=1), # 80 x 60
                         ConvBlock1(8,16,kernel_size=3,stride=2,padding=1), # 40 x 30
                         ConvBlock1(16,16,kernel_size=3,stride=1,padding=1), # 40 x 30
                         ConvBlock1(16,32,kernel_size=3,stride=2,padding=1), # 20 x 15
                         ConvBlock1(32,32,kernel_size=3,stride=1,padding=1), # 20 x 15 x 32
                         ConvBlock1(32,64,kernel_size=3,stride=2,padding=1), # 10 x 8
                         ConvBlock1(64,64,kernel_size=3,stride=1,padding=1), # 10 x 8 x 64 # idx 9 
                         ConvBlock1(64,64,kernel_size=3,stride=2,padding=1), # 5 x 4
                         ConvBlock1(64,64,kernel_size=3,stride=1,padding=1), # 5 x 4 x 64 # idx 11
                         ConvBlock1(64,64,kernel_size=3,stride=2,padding=1), # 3 x 2
                         ConvBlock1(64,64,kernel_size=3,stride=1,padding=1), # 3 x 2 x 64 
                         dw_conv_bn_relu(64,64,(2,3),stride=(1,1),padding=(0,0),use_batch_norm=True) # 1 x 1 x 64 # idx 14          
        ]
          
        # make it nn.Sequential
        submodule_names = ['in_conv','conv1','conv2','conv3','conv4',
                            'conv5','conv6','conv7','conv8','conv9',
                            'conv10','conv11','conv12','conv13']
        self.features = torch.nn.Sequential(OrderedDict(list(zip(submodule_names, self.features))))

        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
