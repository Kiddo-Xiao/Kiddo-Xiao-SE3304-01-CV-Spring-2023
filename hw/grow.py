import cv2
import numpy as np
import os

IMAGE_SIZE = (1024, 768)

REGION_SIZE = 10
HALF_REGION_SIZE = REGION_SIZE // 2

# 预处理：中值滤波去噪
MEDIAN_KERNEL_SIZE = 5
# 后处理:除小区域，并平滑分割边界。
MORPH_KERNEL_SIZE = (20, 20)
BLUR_KERNEL_SIZE = (20, 20)
POST_THRESHOLD = 50

# 区域增长：共生矩阵相似度
REGION_SIMILARITY_THRESHOLD = 0.70
# 区域增长：区域灰度差
GRAY_DIFF_THRESHOLD = 11.5

class Growth:
    @staticmethod
    def preprocess(img):
        blur = cv2.medianBlur(img, MEDIAN_KERNEL_SIZE)
        return blur

    @staticmethod
    def postprocess(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.blur(img, BLUR_KERNEL_SIZE, borderType=cv2.BORDER_REPLICATE)
        img = cv2.threshold(img, POST_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        return img

    @staticmethod
    def in_range(x, y):
        return 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]

    @staticmethod
    def calc_gray(arr, dx, dy, gray_level=16):
        # 计算灰度共生矩阵
        gray = np.zeros((gray_level, gray_level), dtype=np.float32)
        arr = arr / (256 // gray_level)
        for i in range(REGION_SIZE - abs(dy)):
            for j in range(REGION_SIZE - abs(dx)):
                gray[int(arr[i, j]), int(arr[i + dy, j + dx])] += 1
        return gray

    @staticmethod
    def calc_dist_pixel(img, x1, y1, x2, y2):
        # 计算两个像素的距离
        half = REGION_SIZE // 2
        if not Growth.in_range(x1 - half, y1 - half) or \
                not Growth.in_range(x2 - half, y2 - half) or \
                not Growth.in_range(x1 + half, y1 + half) or \
                not Growth.in_range(x2 + half, y2 + half):
            return 0, 0
        xs1, ys1, xs2, ys2 = x1 - half, y1 - half, x2 - half, y2 - half
        xe1, ye1, xe2, ye2 = x1 + half, y1 + half,  x2 + half, y2 + half

        gray_x0 = Growth.calc_gray(img[ys1:ye1+1, xs1:xe1+1], 1, 0)
        gray_y0 = Growth.calc_gray(img[ys1:ye1+1, xs1:xe1+1], 0, 1)
        gray_x1 = Growth.calc_gray(img[ys2:ye2+1, xs2:xe2+1], 1, 0)
        gray_y1 = Growth.calc_gray(img[ys2:ye2+1, xs2:xe2+1], 0, 1)
        sim_x = cv2.compareHist(gray_x0, gray_x1, cv2.HISTCMP_CORREL)
        sim_y = cv2.compareHist(gray_y0, gray_y1, cv2.HISTCMP_CORREL)
        gray = np.mean(img[ys2:ye2+1, xs2:xe2+1])
        return sim_x, sim_y, gray

    def traverse_adjacent_pixel(img, new_img, gray, x, y):
        new_markers = []
        adjacent = [(x + j * REGION_SIZE, y + i * REGION_SIZE) for i in range(-1, 2) for j in range(-1, 2)]

        for i, j in adjacent:
            if not Growth.in_range(i, j) or (i == x and j == y) or new_img[j, i] == 255:
                continue
            s1, s2, g = Growth.calc_dist_pixel(img, x, y, i, j)
            if s1 > REGION_SIMILARITY_THRESHOLD and \
                    s2 > REGION_SIMILARITY_THRESHOLD and \
                    abs(g - gray) < GRAY_DIFF_THRESHOLD:
                cv2.rectangle(new_img, (i - HALF_REGION_SIZE, j - HALF_REGION_SIZE),
                              (i + HALF_REGION_SIZE, j + HALF_REGION_SIZE), 255, -1)
                new_markers.append((i, j))

        return new_img, new_markers

    def region_grow(self, img, seeds):
        result = np.zeros_like(img)
        for seed in seeds:
            marks = [seed]
            gray = [np.mean(img[seed[1]-REGION_SIZE:seed[1]+REGION_SIZE+1,
                                 seed[0]-REGION_SIZE:seed[0]+REGION_SIZE+1])]
            new_img = np.zeros_like(img)
            cv2.rectangle(new_img, (seed[0] - HALF_REGION_SIZE, seed[1] - HALF_REGION_SIZE),
                          (seed[0] + HALF_REGION_SIZE, seed[1] + HALF_REGION_SIZE), 255, -1)
            while len(marks) > 0:
                marker = marks.pop()
                new_img, new_marks = Growth.traverse_adjacent_pixel(img, new_img, gray, marker[0], marker[1])
                marks.extend(new_marks)
            result = cv2.bitwise_or(result, new_img)
        return result


    def grow(self, seeds, img_path, out_path):
        origin_img = cv2.imread(img_path)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        img = self.preprocess(origin_img)
        img = self.region_grow(img, seeds)
        mask = self.postprocess(img)
        mask = cv2.bitwise_and(origin_img, mask)
        cv2.imwrite(out_path, mask)


# 定义鼠标监听事件和回调函数
class MouseEventHandler:
    def __init__(self):
        self.seeds = []
        self.img_path = ""
        self.out_path = ""
        self.growth = Growth()

    def draw_seed(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed = (x, y)
            self.seeds.append(seed)
            cv2.circle(param, seed, 2, (0, 255, 0), -1)

    def segment_image(self, img_path, out_path):
        self.seeds = []
        self.img_path = img_path
        self.out_path = out_path

        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)

        cv2.namedWindow("Segmentation")
        cv2.setMouseCallback("Segmentation", self.draw_seed, img)

        while True:
            cv2.imshow("Segmentation", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("w"):
                return 1

        cv2.destroyAllWindows()

        self.growth.grow(self.seeds, self.img_path, self.out_path)
        return 0


if __name__ == "__main__":
    mouse_handler = MouseEventHandler()
    input_folder = 'hw/data-beforeSeg'
    output_folder = 'hw/data-afterSeg-grow'
    for filename in os.listdir(input_folder):
    # filename = '55.png'
    # filename = '219.png'
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if mouse_handler.segment_image(input_path, output_path) == 1:
            break
