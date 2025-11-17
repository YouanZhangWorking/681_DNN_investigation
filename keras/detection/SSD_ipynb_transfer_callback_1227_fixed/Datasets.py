import tensorflow as tf
from keras.layers import Conv2D, Dense, DepthwiseConv2D,add
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np
import math
import keras
from PIL import Image
from random import shuffle
from keras import layers as KL


def cvtColor(image, cvt2color='grey'):
    if cvt2color == 'grey':
        if len(np.shape(image)) == 3:
            image = image.convert('L')
    if cvt2color == 'rgb':
            image = image.convert('rgb')
    return image


# this part need more attention.

class SSDDatasets(keras.utils.Sequence):
    # train_dataloader = SSDDatasets(train_lines, input_shape, anchor, batch_size, num_cls, train=True)
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, train, overlap_threshold=0.4, imgcolor='grey'):
        self.annotation_lines = annotation_lines  # 读取数据集
        self.length = len(self.annotation_lines)  
        self.input_shape = input_shape             # (120, 160)
        self.anchors = anchors   # [0:1242]: [[0.,0., 0.279.., 0.24...],...]; (1242,4)            
        self.num_anchors = len(anchors)  # 1242
        self.batch_size = batch_size # 1
        self.num_classes = num_classes # 2
        self.train = train # true
        self.overlap_threshold = overlap_threshold  # 0.4
        self.imgcolor = imgcolor # 'grey'
    
    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))  # 向上取整
    
    def __getitem__(self ,index):
        image_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size): # (0,16)
            i = i % self.length # 0~347依次循环
            image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train) 
            
            if len(box) != 0:
                boxes = np.array(box[:,:4], np.float32)
                # 进行归一化，调整到0~1之间
                boxes[:,[0,2]] = boxes[:,[0,2]]/(np.array(self.input_shape[1],np.float32))
                boxes[:,[1,3]] = boxes[:,[1,3]]/(np.array(self.input_shape[0],np.float32))
                one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]  # [0:2] [array([1.]), array([1.])]
                box = np.concatenate([boxes, one_hot_label], axis=-1)
                
            box = self.assign_boxes(box)
            image_data.append(image)
            box_data.append(box)
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = image_data.astype(np.float32) / 127.5 - 1.0  
        box_data = np.array(box_data)
        
        return image_data, box_data
    
    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
               
    def get_random_data(self, annotation_line, input_shape, jitter=.3, random=True):  # jitter颜色相关
        line = annotation_line.split() 
        image = Image.open(line[0])
        image = cvtColor(image, cvt2color=self.imgcolor)
        iw, ih = image.size # [375,500]
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        # [79,281,202,451,0], [106,128,250,297,0]
        
        if not random:  # test
            scale = min(w / iw, h / ih)  
            nw = int(iw * scale)  
            nh = int(ih * scale)  
            dx = (w - nw) // 2  
            dy = (h - nh) // 2  
            
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('L', (w, h))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.uint8)
            
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  

            return image_data, box
        
        new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)  
        scale = self.rand(.25, 2)  # 1.5320
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)  # 245
            nh = int(nw / new_ar)  # 235
        image = image.resize((nw, nh), Image.BICUBIC)
               
        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('L', (w, h)) # w=160,h=120
        new_image.paste(image, (dx, dy))  
        image = new_image
        
        #  翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data = np.array(image, np.uint8)
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box
    
    def iou(self, box):# box=[0.375, 0.25, 0.59, 0.59]
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):# box=[0.375, 0.25, 0.59, 0.59]
        iou = self.iou(box)  # (1242,)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_anchors = self.anchors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()
  
    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0 
        if len(boxes) == 0: 
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask] = 1
        return assignment
