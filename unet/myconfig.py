"""
此文件用于配置模型的参数
"""

import torch

MODEL_PATH = r'./model'
IMG_PATH_NAME = 'images'
LABEL_PATH_NAME = 'labels'

IMG_SIZE = (256, 192)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
BATCH_SIZE = 1

LR = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.99

STEP_SIZE = 5
GAMMA = 0.1

TRAIN_SIZE = 0.8
EVAL_SIZE = 1 - TRAIN_SIZE

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# 后处理
MORPH_KERNEL_SIZE = (29, 29)
BLUR_KERNEL_SIZE = (15, 15)
POST_THRESHOLD = 0.5
