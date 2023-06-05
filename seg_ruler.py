import cv2
import numpy as np
import math
from tool import *
# 展示照片
def cv_show(name, img):
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass



def get_ruler(src):

    # 读取图片

    # 读取图片
    img = cv2.imread(src)

    # cv_show('img', img)
    img_cut = cut_image(img)

    # 显示截取后的图片
    # cv2.imshow('Cut Image', img_cut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    ret1, img10 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#（图像阈值分割，将背景设为黑色）


    # 对灰度图像进行二值化处理
    _, thresh = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY)
    # cv_show('thresh', thresh)
    # 轮廓检测（找到打点的位置）
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    meax_x = 0
    meax_y = 0
    counter = 0
    boundRect = []
    # 遍历每个轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积小于一定阈值，则忽略
        if area < 6 or area > 8:
            continue
        
        # 得到轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 在原图上画出外接矩形
        cv2.rectangle(img_cut, (x, y), (x+w, y+h), (0, 0, 255), 2)
        counter = counter +1 
        meax_x = meax_x + x
        meax_y = meax_y + y
        boundRect.append([x, y])

    # 打印boundRect
    print(boundRect)
    start = boundRect[0][1]
    end = boundRect[counter-1][1]
    ruler_size = (start - end)/counter
    ruler_size_percm = int(ruler_size)
    print("每厘米对应的",ruler_size_percm)


    # 显示结果
    # cv2.imshow("Result", img_cut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ruler_size_percm

if __name__ == '__main__':
    get_ruler('./dataset/images/166.png')

