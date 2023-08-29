import time
import cv2
import numpy as np
from PIL import Image
from ssd_pred_mod import SSD


if __name__ == "__main__":
    ssd = SSD()
    mode = "predict" # or dir_predict  # dir predict用来预测文件夹，predict用来预测文件图片
    crop = False  # 指定了是否在单张图片预测后对目标进行截取
    count = False  # 指定了是否进行目标的计数
    # dir_origin_path= "img/"  # 指定了用于检测的图片的文件夹路径
    # dir_save_path = "img_out/"  # 指定了检测完图片的保存路径
    pred_img_path = "C:/Users/5109I21112/Documents/clone/681_DNN_investigation/keras/detection/dataset/JPEGImages/230317_110152_00003.jpg"
    
    if mode == "predict":
        img = pred_img_path
        try:
            image = Image.open(img)
            print("*******",type(image))
        except:
            print('Open Error! Try again!')
        else:
            r_image = ssd.detect_image(image, crop=crop, count=count)
            # r_image.show()
            r_image.save(r"C:\Users\5109I21112\Documents\clone\681_DNN_investigation\keras\detection\SSD_ipynb_transfer_callback\img.jpg")