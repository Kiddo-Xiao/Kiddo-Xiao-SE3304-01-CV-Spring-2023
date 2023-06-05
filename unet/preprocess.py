import cv2
import numpy as np
from PIL import Image
import os


def cropping(img_ori):
    img = cv2.medianBlur(img_ori, 7)
    height, width = img.shape
    half_height = height // 2
    half_width = width // 2

    right, left, top, bottom = 0, 0, 0, 0

    x = 0
    left_flag, right_flag = False, False
    while True:
        if img[half_height, x] < 2:
            if left_flag and not right_flag and img[half_height, x + 1] < 2:
                right = x - 1
                right_flag = True
        else:
            if not left_flag:
                left = x
                left_flag = True
        x += 1
        if x >= width:
            return img
        if left_flag and right_flag:
            break

    y = height - 1
    top_flag, bottom_flag = False, False
    while True:
        if img[y, half_width] < 2:
            if bottom_flag and not top_flag and img[y - 1, half_width] < 2:
                top = y + 1
                top_flag = True
        else:
            if not bottom_flag:
                bottom = y
                bottom_flag = True
        y -= 1
        if y < 0:
            return img
        if top_flag and bottom_flag:
            break

    img_crop = img_ori[top:bottom, left:right]
    img = np.zeros_like(img)
    img[top:bottom, left:right] = img_crop

    return img
