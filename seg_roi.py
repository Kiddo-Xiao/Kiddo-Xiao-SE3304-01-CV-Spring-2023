import numpy as np
import math
from tool import *
import matplotlib.pyplot as plt
import cv2
import os


# 输入是两张照片的路径，分别是原图和分割图， 以及最后的输出位置的文件路径
def find_roi(src_name, seg_name):
    # 读取原图和分割图
    src = cv2.imread(src_name)
    seg = cv2.imread(seg_name)
    name = os.path.basename(src_name)

    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret1, img10 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#（图像阈值分割，将背景设为黑色）

    # 对灰度图像进行二值化处理
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    # 轮廓检测（找到打点的位置）
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = []
    for contour in contours:
        # 将轮廓的点坐标转换为NumPy数组形式
        contour_points = np.concatenate(contour)

        # 对点坐标按照 x 或 y 坐标进行排序
        # 这里使用 x 坐标进行排序，可以根据需要修改为 y 坐标进行排序
        sorted_indices = np.argsort(contour_points[:, 0])

        # 根据排序后的索引重新排列点坐标
        sorted_points = contour_points[sorted_indices]

        # 将排序后的点坐标添加到结果列表
        sorted_contours.append(sorted_points)

    meax_x = 0
    meax_y = 0
    counter = 0
    boundRect = []
    max_height = 0
    max_width = 0
    pre_point = (0,0)

    # 遍历每个轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积小于一定阈值，则忽略
        if area < 5 or area > 10:
            continue
        
        # 得到轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 计算x,y和boundRect中的点的距离
        ifDelete = False
        for point in boundRect:
            if math.sqrt((point[0]-x)**2 + (point[1]-y)**2) < 25:
                ifDelete = True
                continue
        
        if ifDelete:
            continue
        cv2.rectangle(seg, (x, y), (x+w, y+h), (0, 0, 255), 2)

        
        # if math.sqrt((pre_point[0]-x)**2 + (pre_point[1]-y)**2) < 25:
        #     continue
        # 在原图上画出外接矩形
        
        counter = counter +1 
        meax_x = meax_x + x
        meax_y = meax_y + y
        boundRect.append((x, y))
        pre_point = (x, y)

    if(counter == 0):
        return False,[]

    meax_x = (int)(meax_x/counter)
    meax_y = (int)(meax_y/counter)

    sample_point = boundRect[0]
    all_point_x = boundRect
    all_point_y = boundRect
    # 对all_point进行排序，并找到其中x最大和y最大的点
    all_point_x.sort(key=lambda x:x[0])

    width = int(math.fabs(all_point_x[0][0] - meax_x))
    height = int(math.fabs(all_point_x[1][1] - meax_y))
    # height = sample_point[1]-meax_y
    """
        用来debug画图用的会点出四个点的位置
    """
    cv2.circle(seg, (meax_x-width-10, meax_y-height-10), 5, (0, 255, 255), -1)
    cv2.circle(seg, (meax_x + width +10, meax_y+height+10), 5, (255, 255, 0), -1)
    cv2.circle(seg, (meax_x, meax_y), 5, (255, 255, 255), -1)
    cv2.rectangle(seg, (meax_x-width-10, meax_y-height-10), (meax_x + width +10, meax_y+height+10), (0, 0, 255), 2)

    start_rect = [meax_x-width, meax_y-height]
    end_rect = [meax_x+width, meax_y+height]

    start_rect_x = min(meax_x-width,meax_x+width)
    end_rect_x = max(meax_x-width,meax_x+width)
    start_rect_y = min(meax_y-height,meax_y+height)
    end_rect_y = max(meax_y-height,meax_y+height)

    expand_size = 15

    cropped_image = src_gray[start_rect_y-expand_size:end_rect_y+expand_size, start_rect_x-expand_size:end_rect_x+expand_size].copy()

    size = math.sqrt((sample_point[0]-meax_x)**2 + (sample_point[1]-meax_y)**2)

    # 用中值滤波进行处理
    cropped_image = cv2.medianBlur(cropped_image, 7)
    # 用高斯滤波进行处理
    cropped_image = sharpen(cropped_image)
    cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
    edge_detect = canny(cropped_image)
    return True ,boundRect
            

def _eclipse(p1, p2, p3, p4):
    """
    椭圆拟合
    """
    # 估算结节中心, 这个是没问题的
    center_x = (p1[0] + p2[0] + p3[0] + p4[0]) // 4
    center_y = (p1[1] + p2[1] + p3[1] + p4[1]) // 4

    # 估算长短轴
    a = np.sqrt((p1[0] - center_x) ** 2 + (p1[1] - center_y) ** 2)
    a += np.sqrt((p2[0] - center_x) ** 2 + (p2[1] - center_y) ** 2)
    a /= 2
    b = np.sqrt((p3[0] - center_x) ** 2 + (p3[1] - center_y) ** 2)
    b += np.sqrt((p4[0] - center_x) ** 2 + (p4[1] - center_y) ** 2)
    b /= 2
     # 估算倾斜角度
    angle = _angle(p1, p2)

    return (center_x, center_y), a, b, angle


def _angle(src_point, dst_point):
    """
    计算射线与x轴的夹角（弧度）
    """
    dx = dst_point[0] - src_point[0]
    dy = dst_point[1] - src_point[1]
    if dx == 0:
        if dy > 0:
            return np.pi / 2
        else:
            return 3 * np.pi / 2
    elif dx > 0:
        return np.arctan(dy / dx)
    else:
        return np.pi + np.arctan(dy / dx)

class Point:
    def __init__(self, pos, distance):
        self.pos = pos
        self.dis = distance


def sort_points(point):
    # 求出四个点的中心点
    center_x = (point[0][0] + point[1][0] + point[2][0] + point[3][0]) // 4
    center_y = (point[0][1] + point[1][1] + point[2][1] + point[3][1]) // 4
    # 求出这四个点到中心点的距离
    distance = []
    for i in range(4):
        distance.append(np.sqrt((point[i][0] - center_x) ** 2 + (point[i][1] - center_y) ** 2))
    points = []
    for i in range(4):
        points.append(Point(point[i], distance[i]))
    # 按照distance对points进行降序排序
    points.sort(key=lambda x: x.dis, reverse=True)
    point = []
    for i in range(4):
        point.append(points[i].pos)
    return point
