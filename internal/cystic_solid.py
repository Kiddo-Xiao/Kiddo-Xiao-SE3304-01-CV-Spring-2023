import cv2
import numpy as np

from internal.config import *


class CysticSolidDetector:
    def __init__(self):
        pass

    def _preprocess(self, thyroid, nodule):
        """
        预处理
        """
        thyroid = cv2.medianBlur(thyroid, MEDIAN_BLUR_SIZE_CS)
        nodule = cv2.medianBlur(nodule, MEDIAN_BLUR_SIZE_CS)
        return thyroid, nodule

    def cystic_solid_detect(self, thyroid_mask_path, nodule_mask_path):
        """
        判断结节是囊性还是实性
        """
        thyroid = cv2.imread(thyroid_mask_path, cv2.IMREAD_GRAYSCALE)
        nodule = cv2.imread(nodule_mask_path, cv2.IMREAD_GRAYSCALE)

        # 预处理
        thyroid, nodule = self._preprocess(thyroid, nodule)

        # 计算结节平均灰度与甲状腺平均灰度的比值
        thyroid_mean = np.sum(thyroid) / np.sum(thyroid > 0)
        nodule_mean = np.sum(nodule) / np.sum(nodule > 0)
        ratio = nodule_mean / thyroid_mean
        print(thyroid_mean, nodule_mean, ratio)

        # 分级：0-0.3: 纯囊性 0.3-0.6: 稠厚囊性 0.6-0.9: 实性（低回声） 0.85-1.1: 实性（等回声） >1.1: 实性（高回声）
        if ratio < PURE_CYSTIC:
            return '纯囊性', ratio, nodule_mean
        elif ratio < THICK_CYSTIC:
            return '稠厚囊性', ratio, nodule_mean
        elif ratio < LOW_ECHO:
            return '实性（低回声）', ratio, nodule_mean
        elif ratio < EQUAL_ECHO:
            return '实性（等回声）', ratio, nodule_mean
        else:
            return '实性（高回声）', ratio, nodule_mean
