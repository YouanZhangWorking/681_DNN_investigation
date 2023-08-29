import cv2
import numpy as np
import tensorflow as tf

import time
import cv2
import numpy as np
from PIL import Image
from camera_ssd_pred import SSD


if __name__ == "__main__":
    ssd = SSD()
    mode = "predict" # or dir_predict  # dir predict用来预测文件夹，predict用来预测文件图片
    crop = False  # 指定了是否在单张图片预测后对目标进行截取
    count = False  # 指定了是否进行目标的计数
    # dir_origin_path= "img/"  # 指定了用于检测的图片的文件夹路径
    # dir_save_path = "img_out/"  # 指定了检测完图片的保存路径

    # 打开摄像头
    # cap = cv2.VideoCapture(0)

    # 读取视频流帧
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    
    
    if mode == "predict":
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = frame
            # (480, 640, 3) uint8
            # print(np.shape(img),img.dtype.name,img)
            # <class 'numpy.ndarray'>
            # print(type(img))
            img = Image.fromarray(img)
            img = ssd.detect_image(img, crop=crop, count=count)
            # <class 'PIL.Image.Image'>
            # print(type(img))
            img = np.array(img)

            cv2.imshow('video',img)
            # 按下'q'退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
        
        