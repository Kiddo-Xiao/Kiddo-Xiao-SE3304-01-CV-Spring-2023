import math
import cv2
import numpy as np

from internal.config import *


class UniformityDetector:
    def __init__(self):
        self.dist = None
        pass

    def _preprocess(self, nodule):
        """
        预处理
        """
        nodule = cv2.medianBlur(nodule, MEDIAN_BLUR_SIZE_UNI)
        return nodule

    def _check_inside(self, nodule, point):
        """
        检查某点是否在结节内部，且到结节边缘的距离不小于n个像素
        """
        # 距离变换
        if self.dist is None:
            self.dist = cv2.distanceTransform(nodule, cv2.DIST_L1, REGION_SIZE)
            self.dist = np.uint8(self.dist)

        # 检查该点是否在结节内部
        if self.dist[point[1], point[0]] < REGION_SIZE:
            return False
        return True

    def _calc_glcm(self, arr, dx, dy, gray_level=16):
        """
        计算灰度共生矩阵
        """
        glcm = np.zeros((gray_level, gray_level), dtype=np.float32)
        arr = arr / (256 // gray_level)
        for i in range(REGION_SIZE - abs(dy)):
            for j in range(REGION_SIZE - abs(dx)):
                glcm[int(arr[i, j]), int(arr[i + dy, j + dx])] += 1
        return glcm

    def _calc_feature(self, image, point):
        """
        计算某点处图像的纹理特征
        """
        # 用灰度共生矩阵计算以point为中心，边长为5的区域内图像的纹理特征
        glcm_x = self._calc_glcm(image[point[1] - 2:point[1] + 3, point[0] - 2:point[0] + 3], 1, 0)
        glcm_y = self._calc_glcm(image[point[1] - 2:point[1] + 3, point[0] - 2:point[0] + 3], 0, 1)
        return glcm_x, glcm_y

    def uniformity_detect(self, nodule_path):
        """
        判断结节的均匀性
        """
        nodule = cv2.imread(nodule_path, cv2.IMREAD_GRAYSCALE)

        # 预处理
        nodule = self._preprocess(nodule)

        # 计算结节内部的纹理特征
        x, y = nodule.shape[1], nodule.shape[0]
        glcm_xs = []
        glcm_ys = []
        for i in range(0, x, REGION_SIZE):
            for j in range(0, y, REGION_SIZE):
                if nodule[j, i] > 0 and self._check_inside(nodule, (i, j)):
                    glcm_x, glcm_y = self._calc_feature(nodule, (i, j))
                    glcm_xs.append(glcm_x)
                    glcm_ys.append(glcm_y)
        self.dist = None

        # 计算标准差
        glcm_xs = np.array(glcm_xs)
        glcm_ys = np.array(glcm_ys)
        std_x = np.std(glcm_xs, axis=0)
        std_y = np.std(glcm_ys, axis=0)
        std = math.sqrt(np.mean(std_x) * np.mean(std_y))

        # 分级：0-0.8：均匀；0.08-0.16：较均匀；0.16-0.3：不均匀；0.3-：非常不均匀
        if std < UNI_THRESHOLD:
            return '内质均匀', std
        elif std < UNI_THRESHOLD1:
            return '内质较均匀', std
        elif std < UNI_THRESHOLD2:
            return '内质不均匀', std
        else:
            return '内质非常不均匀', std
