import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import math

def calcification(a, b, nodule_path, edgelabel_path, nodule_mean, markpoint):
    # 结节边缘label
    img3 = cv2.imread(edgelabel_path)
    # 结节图像
    img = cv2.imread(nodule_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    # 设置膨胀卷积核
    kernel2 = np.ones((5, 5), np.uint8)
    # 设置腐蚀卷积核
    kernel3 = np.ones((2, 2), np.uint8)


    # 图像膨胀处理
    erosion = cv2.dilate(img3, kernel2, iterations=4)
    imgs = cv2.bitwise_and(img, erosion)
    grays = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)


    # 顶帽变换突出打点标记和钙化位置
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat2 = cv2.morphologyEx(grays, cv2.MORPH_TOPHAT, kernel)
    # 对打点标记和钙化位置进行二值化处理(根据结节区域的平均灰度)
    _, thresh = cv2.threshold(tophat, nodule_mean, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, kernel3)# 图像腐蚀处理
    _, thresh2 = cv2.threshold(tophat2, nodule_mean, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.erode(thresh2, kernel3)# 图像腐蚀处理

    # 顶帽变换突出打点标记
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel1)
    tophat3 = cv2.morphologyEx(grays, cv2.MORPH_TOPHAT, kernel1)
    # 对打点标记图像进行二值化处理(看效果再判断是否使用平均灰度)
    _, thresh1 = cv2.threshold(tophat1, 120, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.erode(thresh, kernel3)
    _, thresh3 = cv2.threshold(tophat3, 120, 255, cv2.THRESH_BINARY)
    thresh3 = cv2.erode(thresh3, kernel3)


    # 轮廓检测
    # 钙化位置+打点位置
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 打点位置
    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 结晶位置+打点位置
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 打点位置
    contours3, hierarchy3 = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    boundRect = []
    boundRect1 = []
    eacharea = []
    area = 0
    # 找到结晶位置并标记
    for contour2 in contours2:
        x, y, w, h = cv2.boundingRect(contour2)
        flag = False
        for contour3 in contours3:
            x2, y2, w2, h2 = cv2.boundingRect(contour3)
            if abs(x-x2)<=20 and abs(y-y2)<=20:
                flag = True
                break
        for mark in markpoint:
            if abs(x-mark[0])<=20 and abs(y-mark[1])<=20:
                flag = True
                break
        if not flag:
            # 计算疑似结晶区域平均灰度
            area_mean = np.mean(gray[y:(y + h), x:(x + w)])
            # 判断是否是结晶
            if area_mean/nodule_mean<0.9:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                boundRect1.append([x, y])
    number = len(boundRect1)# 结晶数量

    # 找到钙化位置并标记，剔除原图打点标记
    for contour in contours:
        # 得到轮廓的外接矩形
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        flag = False
        flags = False
        for contour1 in contours1:
            # 得到轮廓的外接矩形
            x2, y2, w2, h2 = cv2.boundingRect(contour1)
            if abs(x1-x2)<=20 and abs(y1-y2)<=20:
                flag = True
                break
        for mark in markpoint:
            if abs(x1 - mark[0]) <= 20 and abs(y1 - mark[1]) <= 20:
                flag = True
                break
        if not flag:
            for i in range(number):
                if abs(x1-boundRect1[i][0])<=1 and abs(y1-boundRect1[i][1])<=1:
                    flags = True
                    break
            if not flags:
                cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
                # 计算钙化区域总面积
                area += cv2.contourArea(contour)
                # 每个钙化区域面积
                eacharea.append(cv2.contourArea(contour))
                # 每个钙化的位置
                boundRect.append([x1, y1])
    num = len(boundRect)# 钙化数量

    # 对钙化进行分类
    if (num==0):
        return "无钙化", img, 0, 0
    elif num>=20 and area/(np.pi*a*b)>=0.5:
        return "弥散钙化", img, num, 0
    else:
        count = 0
        count1 = 0
        for i in range(num):
            if eacharea[i]/(np.pi*a*b)>0.1:
                count = count + 1
            else:
                count1 = count1 + 1
        msg = '%d处粗糙钙化, %d处微钙化' % (count, count1)
        return msg, img, count, count1
