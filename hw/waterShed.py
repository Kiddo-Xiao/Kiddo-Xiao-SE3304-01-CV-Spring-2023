import cv2
import numpy as np
import os


IMAGE_SIZE = (1024, 768)

# 预处理:中值滤波，去除噪声。
MEDIAN_KERNEL_SIZE = 5

# 后处理:去除小区域，并平滑分割边界。
MORPH_KERNEL_SIZE = (30, 30)
BLUR_KERNEL_SIZE = (20, 20)
POST_THRESHOLD = 100

class WatershedSegmenter:
    def __init__(self):
        self.img = None
        self.blur = None
        self.marks = None
        self.seeds_fg = []
        self.seeds_bg = []
        self.window_name = "Image"
        self.is_drawing = False

    def preprocess(self, img):
        blur = cv2.medianBlur(img, MEDIAN_KERNEL_SIZE)
        return blur

    def postprocess(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.blur(img, BLUR_KERNEL_SIZE, borderType=cv2.BORDER_REPLICATE)
        img = cv2.threshold(img, POST_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        return img

    def segment(self):
        self.marks = np.zeros(self.blur.shape[:2], dtype=np.int32)
        for seed in self.seeds_fg:
            cv2.drawMarker(self.marks, tuple(seed), 1, cv2.MARKER_TILTED_CROSS)
        for seed in self.seeds_bg:
            cv2.drawMarker(self.marks, tuple(seed), 2, cv2.MARKER_TILTED_CROSS)
        cv2.watershed(self.blur, self.marks)

    def draw_seed(self, event, x, y, flags, param):
        

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.seeds_fg.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_drawing = True
            self.seeds_bg.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.is_drawing = False

    def show_image(self):
        img_copy = self.img.copy()
        for seed in self.seeds_fg:
            cv2.drawMarker(img_copy, seed, (0, 255, 0), cv2.MARKER_TILTED_CROSS)
        for seed in self.seeds_bg:
            cv2.drawMarker(img_copy, seed, (0, 0, 255), cv2.MARKER_TILTED_CROSS)
        cv2.imshow(self.window_name, img_copy)

    def seg_waterShed(self, img_path, out_path):
        self.img = cv2.imread(img_path)
        self.blur = self.preprocess(self.img)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_seed)

    
        while True:
            self.show_image()
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('w'):
                return 1
        self.segment()
        self.seeds_fg.clear()
        self.seeds_bg.clear()
        # 将每个区域用不同的颜色显示在原始图像上
        segmentation = np.zeros_like(self.blur)
        segmentation[self.marks == 1] = 255
        segmentation[self.marks == 2] = 0

        segmentation = self.postprocess(segmentation)
        segmentation = cv2.bitwise_and(self.img, segmentation)
        cv2.imwrite(out_path, segmentation)

        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    segmenter = WatershedSegmenter()
    input_folder = 'hw/data-beforeSeg'
    output_folder = 'hw/data-afterSeg-waterShed'
    for filename in os.listdir(input_folder):
    # filename = '55.png'
    # filename = '170.png'
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if segmenter.seg_waterShed(input_path, output_path) == 1:
            break

