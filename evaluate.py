import keras
import tensorflow as tf
from callbacks import EvalCallback
from Datasets import SSDDatasets
from utils import get_classes
from Anchors import get_anchors
import numpy as np
import sys
import os
import matplotlib
import shutil

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from Datasets import cvtColor
from utils import resize_image
from utils_bbox import BBoxUtility
from utils_map import get_map


class EvalCallback():
    def __init__(self, model_path, input_shape, anchors, class_names, num_classes, val_lines, \
                 map_out_path=".temp_map_out", path='C:/Users/5109I21112/Documents/clone/681_DNN_investigation/keras/detection/dataset/JPEGImages/', 
                 max_boxes=100, confidence=0.5, nms_iou=0.3, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.model_path = model_path
        self.input_shape = input_shape
        self.anchors = anchors
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.map_out_path = map_out_path
        self.path = path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        # ---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------#
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)

        self.maps = [0]
        self.epoches = [0]

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # print(os.path.join(map_out_path, "detection-results/"+image_id+".txt"))
        # image_shape = np.array(np.shape(image)[0:2])
        image_shape = np.array([image.size[1], image.size[0]])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)  # (120, 160)  np.(h,w,c) pil.(w,h,c)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)  # (120, 160)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.array(image_data, dtype='float32'), 0)
        image_data = np.expand_dims(np.array(image_data, dtype='float32'), -1)
        # image_data = image_data / 127.5 - 1.0
        # print('......',(image_data / 127.5 - 1.0).astype('int8'))
        image_data = image_data.astype('float32') -128
        image_data = np.int8(image_data)
        # print('-------',image_data)
        # print('****',image_data.astype('int8'))
        # image_data = image_data.astype('float32')/255.0
        # image_data = np.int8(image_data*127)
        
        import tensorflow as tf
        # 加载模型
        model_path = r"C:\Users\5109I21112\Documents\clone\681_DNN_investigation\keras\detection\SSD_ipynb_transfer_callback\output\20230819\quantized_model_0818.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # 获取输入输出张量索引
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 推理
        interpreter.set_tensor(input_details[0]['index'],image_data)
        interpreter.invoke()

        # 获取输出
        output_data = interpreter.get_tensor(output_details[0]['index'])
        preds = output_data
        
        # print("show dimension")
        # print(np.shape(preds))
        preds = 0.07807575166225433 * (preds + 8)
        # print(preds)

        # -----------------------------------------------------------#
        #   将预测结果进行解码
        # -----------------------------------------------------------#
        results = self.bbox_util.decode_box(preds, self.anchors, image_shape,
                                            self.input_shape, self.letterbox_image, confidence=self.confidence)
        # --------------------------------------#
        #   如果没有检测到物体，则返回原图
        # --------------------------------------#
        if len(results[0]) <= 0:
            return

        top_label   = np.array(results[0][:, 4], dtype = 'int32')
        # top_label = results[0][:, 4]
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        print(top_conf)
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
            # print(predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom)))

        f.close()
        return

    def eval(self):
        if not os.path.exists(self.map_out_path):
            os.makedirs(self.map_out_path)
        if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
        if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
            os.makedirs(os.path.join(self.map_out_path, "detection-results"))
        print("Get map.")


        
        for annotation_line in tqdm(self.val_lines):
            line = annotation_line.split()
            image_id = os.path.basename(line[0]).split('.')[0]
            # print(image_id)
            # print(self.path+line[0][70:])

            
            # ------------------------------#
            #   读取图像并转换成RGB图像
            # ------------------------------#
            image = Image.open(self.path+line[0][70:])
            image = cvtColor(image)
            # image = np.expand_dims(image, axis=-1)
            # ------------------------------#
            #   获得真实框
            # ------------------------------#
            print(line[1:])
            gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            # ------------------------------#
            #   获得预测txt
            # ------------------------------#
            self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

            # ------------------------------#
            #   获得真实框txt
            # ------------------------------#
            with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for box in gt_boxes:
                    left, top, right, bottom, obj = box
                    obj_name = self.class_names[obj]
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

        print("Calculate Map.")
        temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
        self.maps.append(temp_map)
        # self.epoches.append(temp_epoch)

        
        # file_path = self.log_dir
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        #     print(f"Path '{file_path}' created successfully.")
        # else:
        #     print(f"Path '{file_path}' already exists.")
            
        # with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
        #     f.write(str(temp_map))
        #     f.write("\n")

        plt.figure()
        # plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Map %s' % str(self.MINOVERLAP))
        plt.title('A Map Curve')
        plt.legend(loc="upper right")

        # plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
        plt.cla()
        plt.close("all")

        print("Get map done.")
        shutil.rmtree(self.map_out_path)


if __name__ == "__main__":  

    input_shape = [120, 160]  # 输入的尺寸大小
    anchor_size = [32, 59, 86, 113, 141, 168]  # 用于设定先验框的大小，根据公式计算而来；如果要检测小物体，修改浅层先验框的大小，越小的话，识别的物体越小；
    imgcolor = 'grey'  # imgcolor选“rgb” or “grey”, 则处理图像变单通道或者三通道
    cls_name_path = "C:/Users/5109I21112/Documents/clone/681_DNN_investigation/keras/detection/SSD_ipynb_transfer_callback/model_data/voc_classes.txt"  # 导入目标检测类别；
    model_path = "detection/SSD_ipynb_transfer_callback/output/20230819/quantized_model_0818.tflite"
    
    # 1. 获取classes和anchor
    class_names, num_cls = get_classes(cls_name_path)
    num_cls += 1  # 增加一个背景类别

    # 2. 获取anchors, 输出的是归一化之后的anchors
    anchor = get_anchors(input_shape, anchor_size)
    print("type:",type(anchor), "shape:", np.shape(anchor))

    # 加载数据
    # val_annotation_path = 'detection/dataset/ImageSets/Main/val.txt'
    val_annotation_path = 'C:/Users/5109I21112/Documents/clone/681_DNN_investigation/keras/detection/VOC_dataset/2007_val.txt'
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    
    EvalCallback(model_path, input_shape, anchor, class_names, num_cls, val_lines).eval()

    print('18')
    
    
    

   