# import tensorflow as tf
# from keras import optimizers


# def MV2_struct(inputs, expansion_factor=2, output_channels=16, stride=2):
#     # 扩展层
#     expanded_channels = inputs.shape[-1] * expansion_factor
#     x = tf.keras.layers.Conv2D(expanded_channels, (1, 1), padding='same', use_bias=False)(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU(max_value=6)(x)

#     # 深度卷积 (3x3 Depthwise Convolution)
#     x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU(max_value=6)(x)

#     # 线性瓶颈 (1x1卷积，不带激活函数)
#     x = tf.keras.layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)

#     # 判断是否使用跳跃连接
#     if stride == 1 and inputs.shape[-1] == output_channels:
#         x = tf.keras.layers.Add()([inputs, x])

#     return x

# def model_version_mb2():
#     input_layer = tf.keras.layers.Input(shape=(120, 160, 1))
#     x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)
#     # x = MV2_struct(x)
#     x = MV2_struct(x, output_channels=16, stride=1)
#     x = MV2_struct(x, output_channels=32)
#     x = MV2_struct(x, output_channels=32, stride=1)
#     x = MV2_struct(x, output_channels=32)
#     x = MV2_struct(x, output_channels=16)
#     x = MV2_struct(x, output_channels=16)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(64)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dense(3, activation='softmax')(x)
#     model = tf.keras.models.Model(inputs=input_layer, outputs=x)
#     # model.summary()
#     return model


import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers

def MV2_struct(inputs, expansion_factor=2, output_channels=16, stride=2):
    # 获取输入的通道数
    input_channels = K.int_shape(inputs)[-1]
    expanded_channels = input_channels * expansion_factor
    
    # 扩展层
    x = layers.Conv2D(expanded_channels, (1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 深度卷积 (3x3 Depthwise Convolution)
    x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 线性瓶颈 (1x1卷积，不带激活函数)
    x = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 判断是否使用跳跃连接
    if stride == 1 and input_channels == output_channels:
        x = layers.Add()([inputs, x])

    return x

def model_version_mb2():
    input_layer = layers.Input(shape=(120, 160, 1))
    
    # 第一个卷积层，将输入从1通道扩展到3通道
    x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(input_layer)
    
    # 使用MV2模块
    x = MV2_struct(x, output_channels=16, stride=1)
    x = MV2_struct(x, output_channels=32)
    x = MV2_struct(x, output_channels=32, stride=1)
    x = MV2_struct(x, output_channels=32)
    x = MV2_struct(x, output_channels=16)
    x = MV2_struct(x, output_channels=16)
    
    # 最大池化层
    x = layers.MaxPooling2D(2, 2)(x)
    
    # 展平
    x = layers.Flatten()(x)
    
    # 全连接层和BatchNormalization
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    
    # 输出层 (Softmax)
    x = layers.Dense(3, activation='softmax')(x)
    
    # 构建模型
    model = models.Model(inputs=input_layer, outputs=x)
    
    return model
