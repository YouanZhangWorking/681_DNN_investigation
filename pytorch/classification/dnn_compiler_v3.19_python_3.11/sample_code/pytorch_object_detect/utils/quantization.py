'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import torch
import torch.quantization

# Quantization scheme for IMX681
QCONFIG_GLOBAL = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        qscheme=torch.per_tensor_affine,  # for activations
        dtype=torch.quint8
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.FusedMovingAvgObsFakeQuantize,
        qscheme=torch.per_tensor_symmetric,  # for weights
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8
    )
)

def prepare_qat(model,layers_to_fuse):
    """
    Prepare the model for quantization aware training
    Args:
        model: FP32 model
        layers_to_fuse: List of layer sequences to fuse together (conv2d->batch_norm->relu)    
    Returns:
        Quantized model
    """
    # Set model to evaluation mode
    model.eval()

    # fuse conv/bn/relu layers
    model = torch.quantization.fuse_modules(
    model, layers_to_fuse
    )

    # Set qconfig
    model.qconfig = QCONFIG_GLOBAL

    # Insert observers
    model.train()
    model_prepared = torch.quantization.prepare_qat(model)
    model_prepared.train()

    return model_prepared


def load_quantized_model(model, layers_to_fuse, checkpoint_path, device=torch.device('cpu')):
    """
    Loads a quantized model. 
    
    Args:
        model: The base model architecture (unquantized)
        layers_to_fuse:  List of layer sequences to fuse together (conv2d->batch_norm->relu) 
        checkpoint_path: Path to the quantized model state dict
        device: torch.device to load the model on to 
    """
    # Set model to eval mode
    model.eval()

    # Set qconfig
    model.qconfig = QCONFIG_GLOBAL

    model = torch.quantization.fuse_modules(
    model, layers_to_fuse
    )

    model_prepared = torch.quantization.prepare(model)
    model_quantized = torch.quantization.convert(model_prepared)
    
    # Load the pre-quantized state dict
    state_dict = torch.load(checkpoint_path,map_location=device)
    model_quantized.load_state_dict(state_dict)
    
    return model_quantized
