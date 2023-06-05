import cv2
from tool import *
from rec_edge import *
# 读取图像



def angle_detect(roi_path):
    image = cv2.imread(roi_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray != 0] = 255
    gray = dilate(gray)
    # cv_show('gray', gray)
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    # cv_show('edges', edges)


    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # 输出角点数量
    num_corners = len(corners)
    if num_corners > 40:
        print("存在角")
        return True
    else:
        print("不存在角")
        return False


# main函数
if __name__ == '__main__':
    angle_detect("outputs/roi/180.png")

# 轮廓提取
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 遍历轮廓
# # for contour in contours:
# #     # 多边形拟合
# #     epsilon = 0.002 * cv2.arcLength(contour, True)
# #     approx = cv2.approxPolyDP(contour, epsilon, True)

# #     # 判断规则性
# #     print(len(approx))
# #     if len(approx) <= 20:
# #         # 判断是否有倒角
# #         angles = []
# #         for i in range(4):
# #             p1 = approx[i][0]
# #             p2 = approx[(i + 1) % 4][0]
# #             p3 = approx[(i + 2) % 4][0]

# #             vector1 = p1 - p2
# #             vector2 = p3 - p2

# #             dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
# #             norm_product = cv2.norm(vector1) * cv2.norm(vector2)

# #             angle = cv2.fastArccos(dot_product / norm_product)
# #             angles.append(angle)

# #         # 输出结果
# #         if all(angle > 80 for angle in angles):
# #             print("图形规则且有倒角")
# #         else:
# #             print("图形规则但无倒角")
# #     else:
# #         print("图形不规则")

# # # 显示结果图像
# # cv2.imshow("Result", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



# has_corner = False
# for contour in contours:
#     # 进行凸包检测
#     hull = cv2.convexHull(contour)

#     # 判断凸包的边数
#     if len(hull) > 3:
#         has_corner = True
#         break

# # 输出结果
# if has_corner:
#     print("存在角")
# else:
#     print("不存在角")
