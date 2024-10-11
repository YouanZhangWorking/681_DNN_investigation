import os
import tensorflow as tf
import numpy as np
from keras import backend as K
from PIL import Image

K.clear_session()
print("Tensorflow version:", tf.__version__)

# 1. 制作量化数据集
path = r"/home/zhangyouan/桌面/zya/dataset/681/PCScreen_Book_PhoneScreen/train/"
list_dir = os.listdir(path)

labels = {'PcScreen': 0, 'PhoneScreen': 1, 'book': 2}

test_images = []
test_images_link = []
test_labels = []

for i in list_dir:
    path1 = path + i + "/"
    list_label = os.listdir(path1)
    for j in list_label:
        path2 = path1 + j
        test_labels.append(labels[i])
        test_images_link.append(path2)
        test_images_tmp = Image.open(path2)
        test_images_g = test_images_tmp.convert('L')
        test_images_g_resize = test_images_g.resize((160, 120), Image.ANTIALIAS)  # (width, height)
        test_images.append(np.array(test_images_g_resize))

test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("test dataset size: ", np.shape(test_images))
print("test label size: ", np.shape(test_labels))
test_images = np.expand_dims(test_images, axis=-1)
print("teste_images shape:", np.shape(test_images))
print("original image data:", np.max(test_images[0]), np.min(test_images[0]))  # 判断unsigned
test_images = test_images.astype(np.float32) / 255.0
print("after norm: ", np.max(test_images[0]), np.min(test_images[0]))  # 判断0~1

# 2. uint8量化
h5_model = tf.keras.models.load_model(r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_1009_lower_train_trainmore_1_13.h5")

def representative_data_gen():
    for input_value in test_images:
        input_value = np.expand_dims(input_value, axis=0)
        yield [input_value.astype(np.float32)]

# 使用旧版API来进行模型转换
converter = tf.lite.TFLiteConverter.from_keras_model_file(r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_1009_lower_train_trainmore_1_13.h5")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 使用默认的optimizations标记来量化所有固定参数
converter.representative_dataset = representative_data_gen  # 使用浮点回退量化进行转换

# 使用int8量化
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {'input_1': (0., 1.)}  # 根据实际输入的均值和标准差来设置
tflite_model_quant = converter.convert()

# 保存转换后的模型
tflite_name = r"/home/zhangyouan/桌面/zya/NN_net/network/SSD/IMX_681_ssd_mobilenet_git/keras/classification/trained_model/pc_book_phone_1009_tf113.tflite"
with open(tflite_name, 'wb') as f:
    f.write(tflite_model_quant)
