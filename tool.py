import cv2
import numpy as np
# 读取照片并展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 用来进行边缘识别
def canny(img):
    # 边缘识别
    img_canny = cv2.Canny(img, 120, 254)
    return img_canny

# 裁取图片的右边的十分之一
def cut_image(img):
    # 获取图片宽度
    width = img.shape[1]
    # 计算需要截取的宽度
    cut_width = int(width * 0.1)
    # 截取图片的最右边1/10部分
    img_cut = img[:, -cut_width:]
    return img_cut

# 裁取图片的下方的十分之一
def cut_image_bottom(img):
    # 获取图片高度
    height = img.shape[0]
    # 计算需要截取的高度
    cut_height = int(height * 0.1)
    # 截取图片的最下边1/10部分
    img_cut = img[-cut_height:, :]
    return img_cut

# 裁取图片的左方的十分之一
def cut_image_left(img):
    # 获取图片宽度
    width = img.shape[1]
    # 计算需要截取的宽度
    cut_width = int(width * 0.3)
    # 截取图片的最左边1/10部分
    img_cut = img[:, :cut_width]
    return img_cut


# 对图片进行高斯模糊
def gaussian_blur(img):
    # 高斯模糊
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    return img_blur


# 增强灰度图片的对比度
def enhance_contrast(img):
    # 增强对比度
    img_enhance = cv2.equalizeHist(img)
    return img_enhance

# 锐化图片
def sharpen(img):
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 定义卷积核
    img_sharpen = cv2.filter2D(img, -1, kernel=kernel)
    return img_sharpen
