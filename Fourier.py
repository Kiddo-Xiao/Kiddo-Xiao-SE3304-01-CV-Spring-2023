import cv2
import numpy as np
from matplotlib import pyplot as plt


def Fouride(src):
    # 读取图像并转换为灰度图像
    image = cv2.imread('outputs/roi/64.png', 0)

    # 进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 构建高通滤波器
    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # 进行频域滤波
    fshift_filtered = fshift * mask

    # 进行傅里叶逆变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    image_filtered = np.fft.ifft2(f_ishift)
    image_filtered = np.abs(image_filtered)

    # 显示原始图像和滤波后的图像
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_filtered, cmap='gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

    plt.show()
