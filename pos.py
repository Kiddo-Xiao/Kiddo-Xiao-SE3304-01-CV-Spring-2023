import cv2
import numpy as np
import math
from tool import *
import matplotlib.pyplot as plt

# ！！！ VSCode成功运行需要调Python解释器版本为Python3.9.12 ~\anaconda3\python.exe(Conda) ！！！

# 定义函数：用于剪裁图像
def cut_image(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

# 设定阈值，用于过滤匹配度低的区域
threshold_icon = 0.6
threshold_focus = 0.75

# 
def no_icon(seg):
    # image应当且需要为灰度图像
    image = get_label(seg)
    # 用高斯滤波进行处理
    
    image = cv2.GaussianBlur(image, (15, 15), 0)
    # image = sharpen(image)
    # cv2.imshow("sharpen",image)
    # Canny 边缘检测
    # edges = canny(image)
    # cv2.imshow("edge_detect",edges)

    # 轮廓检测:cv2.RETR_EXTERNAL 指定了轮廓检测的模式为仅检测最外层的轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 遍历轮廓并获取边界框
    for contour in contours:
        # print("contour.len = ",len(contour))
        for point in contour:
            cv2.circle(image, tuple(point[0]), 1, (100, 0, 0), -1)
            print(tuple(point[0]))
        x, y, w, h = cv2.boundingRect(contour)
        # 在图像中绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (100, 0, 0), 2)

    # 显示带有边界框的图像
    cv2.imshow("Image with Bounding Boxes", image)

    # 固定 x 坐标值
    fixed_x = [x + (int)(w/5) , x + (int)(w/5)*4 , x + (int)(w/2)] 

    # 输出符合 x 坐标等于固定值的轮廓点的 y 坐标
    y_coordinates = []
    for i in fixed_x:
        # print(i)
        for contour in contours:
            # 筛选符合 x 坐标等于固定值的点的 y 坐标
            selected_points = contour[contour[:, :, 0] == i]
            while len(selected_points) != 2:
                i += 1
                if i >= x + w:
                    print("ERROR:out of range!!!(check_roi_pos)")
                    break
                # print("!!!",i)
                selected_points = contour[contour[:, :, 0] == i]
            y_coordinates.extend(selected_points[:, 1].tolist())
            # 在图像中标记符合条件的点
            for point in selected_points:
                cv2.circle(image, tuple(point), 3, (100, 0, 100), -1)
            #如果识别到多个连通区域 只取一个
            if len(contours)>1 :
                print("ERROR:检测到多个连通区域（no_icon）")
                break

    # 显示带有标记点的图像
    cv2.imshow("Image with Marked Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 输出符合条件的 y 坐标
    # print(f"符合 x 坐标等于 {fixed_x} 的轮廓点的 y 坐标：{y_coordinates}")
    left = abs(y_coordinates[1]-y_coordinates[0])
    right = abs(y_coordinates[3]-y_coordinates[2])
    middle = abs(y_coordinates[5]-y_coordinates[4])
    # print("Left纵径 =",left,"Right纵径 =",right,"Middle纵径 =",middle)
    orient = 0
    if(middle < left/2 or middle < right/2) :
        orient = 1
        result = "切面方向: 横断切面\n"
        if(left < right):
            orient = 1
            result += "左右位置（仅供参考）: 左叶甲状腺（超声图左侧为右甲状腺，右侧为左）"
        else :
            orient = 3
            result += "左右位置（仅供参考）: 右叶甲状腺（超声图左侧为右甲状腺，右侧为左）"
    else :
        orient = 2
        result = "切面方向: 纵断切面\n"
        result += "该图像无法判断左右叶"
    return result,orient






# 输入图片序号,输出小图标切片信息；没有小图标则根据甲状腺形状输出左右(126.png)
def get_icon(src,seg):
    ''' 1. 加载模板图像和待匹配图像 '''
    icon1 = cv2.imread('data-model/icon1.png', 0)
    icon2 = cv2.imread('data-model/icon2.png', 0)
    icon3 = cv2.imread('data-model/icon3.png', 0)
    focus1 = cv2.imread('data-model/focus1.png', 0)
    focus2 = cv2.imread('data-model/focus2.png', 0)
    focus3 = cv2.imread('data-model/focus3.png', 0)
    img = cv2.imread(src,0)

    ''' 2. 提取小图标（图像匹配）'''
    # 使用 TM_CCOEFF_NORMED 方法进行匹配
    res1 = cv2.matchTemplate(img, icon1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img, icon2, cv2.TM_CCOEFF_NORMED)
    res3 = cv2.matchTemplate(img, icon3, cv2.TM_CCOEFF_NORMED)

    # 预设图表类型：0为无图标，1为T型，2为i型
    match_type = 0
    
    # 过滤匹配度低的区域
    loc1 = np.where(res1 >= threshold_icon)
    loc2 = np.where(res2 >= threshold_icon)
    loc3 = np.where(res3 >= threshold_icon)

    ''' 3. 绘制边框并剪裁小图标 '''
    for pt in zip(*loc1[::-1]):
        x1, y1 = pt
        x2, y2 = x1 + icon1.shape[1], y1 + icon1.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        match_type = 1
    for pt in zip(*loc2[::-1]):
        x1, y1 = pt
        x2, y2 = x1 + icon2.shape[1], y1 + icon2.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        match_type = 2
    for pt in zip(*loc3[::-1]):
        x1, y1 = pt
        x2, y2 = x1 + icon3.shape[1], y1 + icon3.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        match_type = 1
    # cv2.imshow('Result', img)

    ''' ！！！没有小图标！！！ '''
    if match_type == 0 :
        return no_icon(seg)
    
    icon_img = cut_image(img, x1, y1, x2, y2)
    # cv2.imshow('Icon Image', icon_img)

    ''' 4. 处理小图标 '''
    # 定义旋转角度和旋转中心
    angle_range = range(0, 360, 10)

    if match_type == 1 :
            focus = focus1
    if match_type == 2 :
            focus = focus2
    for angle in angle_range:
        if match_type == 0 :
            break
        center = (focus.shape[1] // 2, focus.shape[0] // 2)

        # 计算旋转后的图像大小，并计算平移矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_width = int((focus.shape[0] * sin) + (focus.shape[1] * cos))
        new_height = int((focus.shape[0] * cos) + (focus.shape[1] * sin))
        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        # 进行旋转变换
        foces_rotated = cv2.warpAffine(focus, M, (new_width, new_height))
        res = cv2.matchTemplate(icon_img, foces_rotated, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold_focus)

        # 矩形框出小图标上的病灶
        for pt in zip(*loc[::-1]):
            x1, y1 = pt
            x2, y2 = x1 + foces_rotated.shape[1], y1 + foces_rotated.shape[0]
            cv2.rectangle(icon_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        if (res >= threshold_focus).any() :
            cv2.imshow('icon_img', icon_img)
            # 输出切面方向和病灶左右位置信息
            orient = 0
            if angle <= 45 or (angle >= 135 and angle <= 225) or angle >=315:
                orient = 2
                result = "切面方向: 纵断切面\n"
            else :
                orient = 1
                result = "切面方向: 横断切面\n"
            if (x1+x2)/2 >= icon_img.shape[0]/2 :
                orient = 1
                result += "左右位置: 左叶甲状腺（超声图左侧为右甲状腺，右侧为左）"
            else :
                orient = 3
                result += "左右位置: 右叶甲状腺（超声图左侧为右甲状腺，右侧为左）"

    # cv2.destroyAllWindows()
    return result,orient

#二值化将甲状腺seg转为label格式
def get_label(seg):
    print("由seg获取label： "+seg)
    seg = cv2.imread(seg,0)
    _, binary = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow("label",binary)
    # 先膨胀后腐蚀将白色甲状腺区域中的黑色roi打点标记去除
    kernel = np.ones((3, 3), np.uint8)# 定义结构元素，用于形态学操作
    dilated = cv2.dilate(binary, kernel, iterations=1)# 膨胀操作，扩展白色区域
    eroded = cv2.erode(dilated, kernel, iterations=1)# 腐蚀操作，收缩白色区域
    mask = cv2.bitwise_xor(binary, eroded)# 生成遮罩
    # cv2.imshow("mask",mask)
    result = cv2.bitwise_or(binary, mask)# 应用遮罩，将黑点抹为白色
    
    # 先膨胀后腐蚀将黑色背景区域中的白色杂点去除(图片灰度颜色取反重复上述操作再取反)
    result = cv2.bitwise_not(result)
    dilated = cv2.dilate(result, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    mask = cv2.bitwise_xor(result, eroded)
    result = cv2.bitwise_or(result, mask)
    result = cv2.bitwise_not(result)
    # cv2.imshow("label",result)
    return result

# 输入是roi_label的位置，输出roi中心坐标x,y,w,h
def get_roi(roi_label):
    # roi = get_label(roi_label)
    roi = cv2.imread(roi_label,0)
    roi = cv2.GaussianBlur(roi, (9, 9), 0)
    # image = sharpen(image)
    # cv2.imshow("sharpen",image)

    # 轮廓检测:cv2.RETR_EXTERNAL 指定了轮廓检测的模式为仅检测最外层的轮廓
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 遍历轮廓并获取边界框
    for contour in contours:
        # print("contour.len = ",len(contour))
        # for point in contour:
            # cv2.circle(image, tuple(point[0]), 1, (100, 0, 0), -1)
            # print(tuple(point[0]))
        x, y, w, h = cv2.boundingRect(contour)
        # 在图像中绘制边界框
        cv2.rectangle(roi, (x, y), (x + w, y + h), (100, 0, 0), 2)
        roi_centerX , roi_centerY = x+w/2 , y+h/2
        # cv2.imshow("",roi)
        if len(contours) > 1:
            print("ERROR:检测到多个连通区域（get_roi）")
        return roi_centerX , roi_centerY,w,h

# 获取roi相对位置信息
def check_roi_pos(seg,roi_label,orient,roi_wSize,roi_hSize,have_OCR):
    image = get_label(seg)
    # 用高斯滤波进行处理
    
    image = cv2.GaussianBlur(image, (15, 15), 0)
    # image = sharpen(image)
    # cv2.imshow("sharpen",image)
    # 轮廓检测:cv2.RETR_EXTERNAL 指定了轮廓检测的模式为仅检测最外层的轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 遍历轮廓并获取边界框
    for contour in contours:
        # print("contour.len = ",len(contour))
        # for point in contour:
            # cv2.circle(image, tuple(point[0]), 1, (100, 0, 0), -1)
            # print(tuple(point[0]))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (100, 0, 0), 2)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        leftX , rightX = x+w/3 , x+w*2/3
        if len(contours) > 1:
            print("ERROR:检测到多个连通区域（check_roi_pos）")
        break;
    
    # 获取病灶中心位置坐标和图像处理获得的大致宽、高
    roiX , roiY , roiW , roiH= get_roi(roi_label)
    roiX = (int)(roiX)
    # 输出符合x坐标等于固定点x且距离目标点位置最近的轮廓点的y坐标
    y_coordinates = []
    for contour in contours:
        # 筛选符合 x 坐标等于固定值的点的 y 坐标
        selected_points = contour[contour[:, :, 0] == roiX]
        while len(selected_points) < 2:
            roiX += 1
            if roiX >= x + w:
                print("ERROR:out of range!!!(check_roi_pos)")
                break
            selected_points = contour[contour[:, :, 0] == roiX]
        min_maxY = y+h # roi下方最近的y
        max_minY = y # roi上方最近的y
        for y in y_coordinates:
            if(y <= min_maxY and y >= roiY):
                min_maxY = y
            if(y >= max_minY and y <= roiY):
                max_minY = y
        for point in selected_points:
            cv2.circle(image, tuple(point), 3, (100, 0, 100), -1)
        #如果识别到多个连通区域 只取一个
        if len(contours)>1 :
            break
    
    if(roiY-max_minY < min_maxY-roiY):
        result = "表面/背面：表面\n"
    else :
        result = "表面/背面：背面\n"

    
    if (orient == 2):#纵断
        if(roiX<leftX):
            result += "上/中/下：上\n"
        elif(roiX>rightX):
            result += "上/中/下：下\n"
        else:
            result += "上/中/下：中\n"
    elif(orient == 1):#横断且左叶（物理的右侧）
        if roiX < rightX:
            result += "内侧/外侧：内侧\n"
        else :
            result += "内侧/外侧：外侧\n"
    elif (orient == 3):#横断且右叶（物理的左侧）
        if roiX <= leftX:
            result += "内侧/外侧：外侧\n"
        else :
            result += "内侧/外侧：内侧\n"
    else :# 未获得切面方向
        print("ERROR:未获得切面方向(check_roi_pos)")
    
    # 判断水平/垂直位：优先用OCR数值判断
    if(have_OCR==True):
        if(roi_wSize > roi_hSize):
            result += "平行位（结节的最长轴平行于皮肤）【无恶性风险】\n"
        else :
            result += "垂直位（结节的最长轴垂直于皮肤）【有恶性风险】\n"
    else :
        if(roiW > roiH):
            result += "平行位（结节的最长轴平行于皮肤）【无恶性风险】\n"
        else :
            result += "垂直位（结节的最长轴垂直于皮肤）【有恶性风险】\n"

    return result

# 定义主函数
if __name__ == '__main__':
    # src = "dataset/images/213.png"    
    # seg = "dataset/seg/213.png"    
    # get_icon(src,seg)
    src = "dataset/images/IM_0186.jpg"  
    seg = "dataset/seg/IM_0186.jpg"    
    result,orient = get_icon(src,seg)
    print(result)
    orient = 2
    seg = "dataset/imagesTr/THYROID_062_0000.png"    
    roi = seg[:8] + "test/" + seg[17:-9] + seg[-4:]
    print("seg : ",seg," | roi : ",roi)

    print(check_roi_pos(seg,roi,orient,1,1,0))

 