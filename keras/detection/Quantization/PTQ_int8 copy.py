import numpy as np
import tensorflow as tf
import os
from PIL import Image
from keras.models import load_model
# 模型训练是设定了图片归一化到(-1,1),方法是：img/127.5-1.0

h5_model_path = "/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand_yxhw_third_screen/output/pc_screen/20240810_pc_screen.h5"
imgs_path = "/home/zhangyouan/桌面/zya/dataset/681/PC_Screen_D/VOCdevkit/VOC2007/JPEGImages/"


# 设置转换项： 配置转换选项，包括输入数据的数据类型(例如float32和uint8)和优化选项。在这里，需要将激活和权重量化为int8
keras_model = load_model(h5_model_path, custom_objects={"compute_loss":None})
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 设置优化选项
converter.target_spec.supported_types = [tf.int8]  # 将权重和激活量化为int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 1. 准备数据，将这些数据用于校准模型，以确定量化参数；
all_img_link = []
all_img = []
list_dir = os.listdir(imgs_path)
for i in list_dir:
    img_path = imgs_path + i
    all_img_link.append(img_path)
    all_img_tmp = Image.open(img_path).convert('L')  # 转为灰度
    all_img_tmp = all_img_tmp.resize((160, 120), Image.ANTIALIAS)  # (width, height)
    all_img.append(np.array(all_img_tmp))

input_data = all_img
input_data = np.array(input_data)
input_data = np.expand_dims(input_data, axis=-1)
# input_data = input_data.astype(np.float32)
input_data = input_data.astype(np.float32) / 127.5 - 1.0 

print("test dataset size: ", np.shape(input_data))

#    校准模型，使用校准数据来估计量化参数
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(input_data).batch(1).take(90):
        yield [input_value]
converter.representative_dataset = representative_data_gen


# 2. 将模型转换为TFLite模型，执行转换操作，并将量化的TFLite模型保存为文件；
tflite_model = converter.convert()
#    保存为TFLite文件
with open('/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/detection/SSD_ipynb_transfer_callback_1227hand_yxhw_third_screen/20240810_pc_screen.tflite', 'wb') as f:
    f.write(tflite_model)
    print("has been written to: 20240810_pc_screen.tflite")
