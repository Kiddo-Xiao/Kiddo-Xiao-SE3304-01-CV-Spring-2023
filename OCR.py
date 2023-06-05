import cv2
import pytesseract
from tool import *
import re
import seg_ruler


def extract_numbers(input_string):
    # 提取数字部分
    pattern = r"([\d.]+)"
    matches = re.findall(pattern, input_string)

    # 转换为浮点数并返回结果
    numbers = [float(match) for match in matches]
    return numbers

def process_string(input_string):
    # 去除换行符和空格
    input_string = input_string.replace('\n', '').replace(' ', '')

    # 提取数字、'cm'部分和小数点部分
    pattern = r"([\d.]+cm)"
    matches = re.findall(pattern, input_string)
    processed_string = ' '.join(matches)

    return processed_string


def get_size(src):
    ruler = cut_image_bottom(src)
    # 对ruler进行阈值化
    gray = cv2.cvtColor(ruler, cv2.COLOR_BGR2GRAY)
    ret1, ruler = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # cv_show('ruler', ruler)

    text = pytesseract.image_to_string(gray)
    # print(text)
    text = process_string(text)
    numbers = extract_numbers(text)
    # 对numbers进行切片，如果其中的某个元素等于4.5则将其切除，只保留剩下的元素
    filtered_numbers = [num for num in numbers if num != 4.5 and num != 4.0 and num != 3.5 and num != 2.5]
    # print(filtered_numbers)
    if text == "":
        return False, filtered_numbers
    return True, filtered_numbers


def get_size_interface(src_path):
    src = cv2.imread(src_path)
    result , size = get_size(src)
    if(result == False):
        percm = seg_ruler.get_ruler(src_path)
        print ("未检测到OCR尺寸,后续进行像素点标定,像素比例尺为")
        return False, percm
    else:
        print(size)
        return True,size



# 定义主函数
if __name__ == '__main__':
    # 输入照片然后进行分割
    path = './dataset/images/57.png'
    get_size_interface(path)
 
