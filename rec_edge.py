import cv2
import numpy as np
import matplotlib.pyplot as plt
import Laplacian as lap
import os
# 展示一张图片，根据路径来展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Fourier(src):
    # 转换为灰度图像
    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


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



# 对图片进行膨胀
def dilate(img):
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    img_dilate = cv2.dilate(img, kernel)
    return img_dilate

# 对照片进行边缘识别
def canny(img):
    # 边缘识别
    img_canny = cv2.Canny(img, 120, 254)
    return img_canny

# 利用中值滤波去除噪点
def medianBlur(img):
    # 中值滤波
    img_medianBlur = cv2.medianBlur(img, 7)
    return img_medianBlur


# 利用傅里叶变换提取边缘
def fourier(img):
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 取绝对值：将复数变化成实数取对数的目的为了将数据变化到较小的范围（比如0-255）
    # 取对数的目的为了将数据变化到较小的范围（比如0-255）
    magnitude_spectrum = np.log(np.abs(fshift))
    return magnitude_spectrum


# 进行梯度分析
def ana_gradient(img):
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    # cv2.imshow('Gradient', gradient)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mean_magnitude = np.mean(gradient)
    variance_magnitude = np.var(gradient)
    print("平均灰度",mean_magnitude)
    print("方差",variance_magnitude)
    return mean_magnitude, variance_magnitude


# 获得轮廓
def get_frame(label):
    # 灰度的形式读入图片
    label = cv2.imread(label)
    label[label!=0]=255
    
    # 转成灰度图片
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)


    # cv_show('label', label)
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #遍历轮廓并绘制方框
    expand_size = 10
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        return w,h

# 分析边缘，要传入src和边缘
def analyze_edge(path_fore,path_roi):
    print(os.path.basename(path_roi))
    # src = cv2.imread(path_src)
    fore = cv2.imread(path_fore)
    roi = cv2.imread(path_roi)
    # cv_show('roi', roi)

    tmp = roi.copy()
    tmp[tmp!=0]=255
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    tmp = dilate(tmp)

    edges = canny(tmp)
    edges = dilate(edges)
    # cv_show('edges', edges)


    expand_roi = cv2.bitwise_and(fore, fore, mask=edges)
    expand_roi = medianBlur(expand_roi)
    expand_roi = cv2.cvtColor(expand_roi, cv2.COLOR_BGR2GRAY)
    
    # expand_roi = cv2.equalizeHist(expand_roi)
    # expand_roi = medianBlur(expand_roi)
    # cv_show('expand_roi', expand_roi)
    # Fourier(expand_roi)
    gradi_result = ana_gradient(expand_roi)
    lap_resule = lap.ana_Laplacian(expand_roi)
    socre = 0
    

    if lap_resule > 0.2:
        socre +=1
    if gradi_result[0] > 0.5:
        socre +=1
    if gradi_result[1] < 90:
        socre +=1
    
    if socre == 0:
        print("边缘不清晰")
        return "边缘不清晰",1
    elif socre == 1:
        print("边缘较清晰")
        return "边缘较清晰",0
    else:
        print("边缘清晰")
        return "边缘清晰",0
    





# main函数
if __name__ == '__main__':
    # pic = '073'
    # path_fore = 'for_edge_detect/ForeTr/THYROID_'+pic+'_0000.png'
    # path_roi = 'for_edge_detect/0-1_roi/THYROID_'+pic+'.png'  
    # analyze_edge(path_fore,path_roi)


    get_frame("./outputs/roi/label/74.png")

