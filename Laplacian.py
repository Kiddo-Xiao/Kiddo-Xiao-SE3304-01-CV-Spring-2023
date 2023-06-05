import cv2
import numpy as np


def ana_Laplacian(src):
    # 转化为灰度图像
    image = src

    # 应用Laplacian算子
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 计算边缘强度
    edge_strength = np.mean(np.abs(laplacian))

    # 打印边缘强度
    print("边缘强度:", edge_strength)
    return edge_strength
