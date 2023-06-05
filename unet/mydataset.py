"""
自定义数据集
"""
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
import os

from unet.myconfig import *


class MyDataset(data.Dataset):
    def __init__(self, img_path, label_path):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.images = os.listdir(self.img_path)
        self.labels = os.listdir(self.label_path)

    def __getitem__(self, index):
        # 读取图片和标签
        img = Image.open(os.path.join(self.img_path, self.images[index]))
        img = img.convert("L")
        img = img.resize(IMG_SIZE)
        label = Image.open(os.path.join(self.label_path, self.labels[index]))
        label = label.convert("L")
        label = label.resize(IMG_SIZE)
        # 将图片和标签转换为tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)
        return img, label

    def __len__(self):
        return len(self.images)
