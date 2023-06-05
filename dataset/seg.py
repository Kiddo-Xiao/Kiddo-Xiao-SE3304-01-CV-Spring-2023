import cv2
import os
# 读取images文件夹中的图片作为src，读取labels文件夹中的图片作为label,label作为mask，将src和lable进行按位与操作，得到的结果保存在merge文件夹中
def seg(src,label):
    src_list = os.listdir(src)
    label_list = os.listdir(label)
    for i in range(len(src_list)):
        src_path = os.path.join(src,src_list[i])
        label_path = os.path.join(label,label_list[i])
        src_img = cv2.imread(src_path)
        label_img = cv2.imread(label_path)
        merge_img = cv2.bitwise_and(src_img,label_img)
        cv2.imwrite('./seg/'+src_list[i],merge_img)

seg('./images','./labels')