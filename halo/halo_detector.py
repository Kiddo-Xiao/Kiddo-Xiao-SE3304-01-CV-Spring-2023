import cv2
import numpy as np

from halo.config import *


class HaloDetector:
    def __init__(self):
        self.width = 0
        self.height = 0

    def _preprocess(self, roi):
        """
        预处理
        """
        self.height, self.width = roi.shape[:2]
        roi = cv2.medianBlur(roi, MEDIAN_BLUR_SIZE)
        roi = cv2.pyrMeanShiftFiltering(roi, PYR_MEAN_SHIFT_FILTERING_SP, PYR_MEAN_SHIFT_FILTERING_SR)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi

    def _angle(self, src_point, dst_point):
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

    def _eclipse(self, p1, p2, p3, p4):
        """
        椭圆拟合
        """
        # 估算结节中心
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
        angle = self._angle(p1, p2)

        return (center_x, center_y), a, b, angle

    def _intersact_eclipse(self, a, b, angle, direction):
        """
        计算射线与椭圆的交点到中心的距离
        """
        delta_angle = direction - angle
        c = np.cos(delta_angle)
        s = np.sin(delta_angle)
        sqr = a * a * s * s + b * b * c * c
        return a * b / np.sqrt(sqr)

    def _grad_detect(self, roi, x0, y0, a, b, angle, direction):
        """
        梯度检测
        """
        # 计算梯度
        dx = np.cos(direction)
        dy = np.sin(direction)
        dist = self._intersact_eclipse(a, b, angle, direction)
        i = 0
        last = None
        grads = []
        while True:
            x = x0 + round(dx * i)
            y = y0 + round(dy * i)
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break
            pixel = int(roi[y, x])
            if last is not None:
                grad = pixel - last
                if RANGE_MIN < i - dist < RANGE_MAX and abs(grad) > GRAD_VALID_THRESHOLD:
                    grads.append(grad)
                else:
                    grads.append(0)
            last = pixel
            i += 1

        # 检测梯度变化，找到晕环的上升沿和下降沿
        sum_group = 0
        descend = []
        ascend = []
        descend_start_flag = False
        descend_end_flag = False
        ascend_start_flag = False
        ascend_end_flag = False
        for i, grad in enumerate(grads):
            if i >= GRAD_GROUP_SIZE - 1:
                sum_group = sum_group + grad - grads[i - GRAD_GROUP_SIZE + 1]
            if not descend_end_flag and sum_group < -GRAD_GROUP_THRESHOLD:
                if not descend_start_flag:
                    descend_start_flag = True
                descend.append(i)
            elif not ascend_end_flag and sum_group > GRAD_GROUP_THRESHOLD:
                if not ascend_start_flag:
                    ascend_start_flag = True
                ascend.append(i)
            else:
                if descend_start_flag:
                    descend_end_flag = True
                if ascend_start_flag:
                    ascend_end_flag = True

        if len(descend) == 0 or len(ascend) == 0 or descend[0] > ascend[0]:
            return None, None

        # 返回晕环的上升沿起点和下降沿起点
        start = round(sum(descend) / len(descend))
        end = round(sum(ascend) / len(ascend))
        start_p = (x0 + round(dx * start), y0 + round(dy * start))
        end_p = (x0 + round(dx * end), y0 + round(dy * end))
        return start_p, end_p

    def halo_detect(self, roi_path, p1, p2, p3, p4):
        print(p1, p2, p3, p4)
        roi = cv2.imread(roi_path)
        roi_gray = self._preprocess(roi)
        center, a, b, angle = self._eclipse(p1, p2, p3, p4)
        print(center, a, b, angle)
        cv2.circle(roi, center, 2, (0, 255, 255), -1)

        # 通过中心向外发散的射线检测晕环边沿
        hasHalo = [0 for _ in range(0, 360, DELTA_ANGLE * GROUP_SIZE)]
        thickness = []
        for i in range(0, 360, DELTA_ANGLE):
            rad = np.deg2rad(i)
            start, end = self._grad_detect(roi_gray, center[0], center[1], a, b, angle, rad)
            if start is not None:
                hasHalo[i // (DELTA_ANGLE * GROUP_SIZE)] = 1
                cv2.line(roi, start, end, (255, 255, 255), 2)
                thickness.append(np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2))

        # 未检测到晕环
        if sum(hasHalo) == 0:
            return False, False, False, roi

        # 判断晕环是否完整
        isComplete = False
        if sum(hasHalo) == len(hasHalo):
            isComplete = True

        # 判断晕环厚度是否均匀
        isEven = False
        if len(thickness) > MIN_VALID_THRESHOLD:
            mean = np.mean(thickness)
            std = np.std(thickness)
            print(mean, std)
            if std < mean * DEVIATION_THRESHOLD:
                isEven = True

        return True, isComplete, isEven, roi


if __name__ == '__main__':
    halo_detector = HaloDetector()
    halo_detector.halo_detect('./rois/180.png', (10, 105), (179, 66), (80, 7), (104, 153))
    # halo_detector.halo_detect('./rois/213.png', (9, 25), (175, 72), (87, 6), (68, 105))
    # halo_detector.halo_detect('./rois/78.png', (10, 59), (203, 85), (141, 8), (107, 133))
    # center, a, b, angle = halo_detector.eclipse((9, 25), (175, 72), (87, 6), (68, 105))
    # print(center, a, b, angle)
