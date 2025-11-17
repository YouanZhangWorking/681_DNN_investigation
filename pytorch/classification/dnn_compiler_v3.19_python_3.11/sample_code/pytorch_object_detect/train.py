'''
Copyright 2025 Sony Semiconductor Solutions Corporation. This is UNPUBLISHED PROPRIETARY SOURCE
CODE of Sony Semiconductor Solutions Corporation. No part of this file may be copied, modified, 
sold, and distributed in any form or by any means without prior explicit permission in writing 
of Sony Semiconductor Solutions Corporation.
'''

import sys
import os
import logging
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.misc import Timer
from utils.quantization import prepare_qat
from model.ssd import MatchPrior
from model.generic_ssdlite import create_generic_ssdlite
from model.config import generic_ssd_config
from utils.misc import set_random_seed
from datasets.custom_images import *
from model.multibox_loss import MultiboxLoss
import yaml
import shutil
import copy

logging.basicConfig(filename='train.log', level=logging.INFO) # log to file
logging.getLogger().addHandler(logging.StreamHandler()) # also print to console output

def train(loader, model, criterion, optimizer, device, debug_steps=100, epoch=-1):
    '''
    Trains the model for one epoch
    '''
    model.train()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes) 
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            
            # Since i == 0 is skipped, we need to add 1 to the divisor
            running_loss_divisor = debug_steps+1 if i == debug_steps else debug_steps

            avg_loss = running_loss / running_loss_divisor
            avg_reg_loss = running_regression_loss / running_loss_divisor
            avg_clf_loss = running_classification_loss / running_loss_divisor
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Loss: {loss.item()}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def test(loader, model, criterion, device):
    '''
    Measure validation loss
    '''
    model.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations = model(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    num = max(num, 1)
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

if __name__ == '__main__':
    set_random_seed(0)

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <filepath>")
    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError(f"File not found: {sys.argv[1]}")
    
    train_cfg_path = sys.argv[1]

    # Load config params
    with open(train_cfg_path,'r') as f:
        params = yaml.safe_load(f)['params']
        dataset_path=params['dataset_path']
        validation_dataset=params['validation_dataset']
        checkpoint_folder=params['checkpoint_folder']
        annotation_format=params['annotation_format']
        label_file=params['label_file']
        batch_size=params['batch_size']
        num_workers=params["num_workers"]
        base_net_lr=params["base_net_lr"]
        extra_layers_lr=params["extra_layers_lr"]
        base_net=params["base_net"]
        use_cuda=params["use_cuda"]
        lr=params["lr"]
        momentum=params["momentum"]
        weight_decay=params["weight_decay"]
        num_epochs=params["num_epochs"]
        validation_epochs=params["validation_epochs"]
        debug_steps=params["debug_steps"]
        t_max=params["t_max"]
        quantize_after = params["quantize_after"]

    # Create the output directory if it doesn't exist
    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    
    loss_log_f = os.path.join(checkpoint_folder,'val_loss_log.txt')
    with open(loss_log_f,'w') as fi:
        fi.write('Epoch,Loss,RegressionLoss,ClassifierLoss\n')

    # Copy the config file to save directory:
    shutil.copyfile(train_cfg_path,os.path.join(checkpoint_folder,'train_config.yaml'))

    # Set up GPU if available, otherwise use CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    timer = Timer()

    # set up
    config = generic_ssd_config

    # Get the models prior boxes
    priors = config.priors

    # object classes
    class_names = tuple([name.strip() for name in open(label_file).readlines()])

    # data transforms
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    # prep training data
    logging.info("Prepare training datasets.")
    datasets = []
    dataset = CustomImagesDataset(dataset_path, transform=train_transform,
                 target_transform=target_transform, is_yolo = True, class_names= class_names, is_gray=True)
    datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    num_classes = len(dataset.class_names)
    print("Num of classes: {}".format(num_classes))
    print("Num of training examples: {}".format(len(dataset.ids)))
    logging.info(train_dataset)
    
    # prep validation data
    logging.info("Prepare Validation datasets.")
    if annotation_format == 'voc':        
        valid_dataset = CustomImagesDataset(validation_dataset, transform=test_transform,
                         target_transform=target_transform, is_yolo=False, class_names= class_names, is_gray=True)
    elif annotation_format == 'yolo':
        valid_dataset = CustomImagesDataset(validation_dataset, transform=test_transform,
                         target_transform=target_transform, is_yolo=True, class_names= class_names, is_gray=True)
    else:
        raise ValueError(f"Annotation Format {annotation_format} is not supported.")
    
    logging.info(valid_dataset)
    logging.info("validation dataset size: {}".format(len(valid_dataset)))
    val_loader = DataLoader(valid_dataset, batch_size,
                            num_workers= num_workers,
                            shuffle=False, drop_last=True)
    
    # construct model
    logging.info("Building network...")
    model = create_generic_ssdlite(num_classes, quantize=True)
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = base_net_lr if base_net_lr is not None else lr
    extra_layers_lr = extra_layers_lr if extra_layers_lr is not None else lr

    timer.start("Load Model")
    model.init()

    # set up optimizer
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    if use_cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids = [DEVICE])

    model.to(DEVICE)    

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    
    logging.info(f"Learning rate: {lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    logging.info("Uses CosineAnnealingLR scheduler.")

    scheduler = CosineAnnealingLR(optimizer, t_max)
    
    # training loop
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    model_quantized=False

    for epoch in range(last_epoch + 1, num_epochs):

        print(f"Starting epoch: {epoch}")
        scheduler.step()
        
        if epoch == quantize_after:
            # Switch to QAT after N epochs
            print("Switching to QAT")
            model = prepare_qat(model.module,config.fuse_layer_list)
            model_quantized=True
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1*lr, momentum=momentum,
            #                 weight_decay=weight_decay)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, t_max)
            
        train(train_loader, model, criterion, optimizer,
              device=DEVICE, debug_steps=debug_steps, epoch=epoch)

        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, model, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )

            # Write to log file
            with open(loss_log_f,'a') as fi:
                fi.write(f'{epoch},{val_loss},{val_regression_loss},{val_classification_loss}\n')

        # save intermediate checkpoints
        model_path = os.path.join(checkpoint_folder, f"Epoch-{epoch}-Quantized_{model_quantized}.pth")

        if model_quantized:
            model.eval()
            model_cpu = copy.deepcopy(model).cpu()
            model_converted = torch.quantization.convert(model_cpu)
            torch.save(model_converted.state_dict(),model_path)
            model.train()
        else:
            if use_cuda and torch.cuda.is_available():
                model.module.save(model_path)
            else:
                model.save(model_path)

        logging.info(f"Saved model {model_path}")
