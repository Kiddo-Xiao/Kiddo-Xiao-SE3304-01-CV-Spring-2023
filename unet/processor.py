import torch.optim
from torch.utils.data import DataLoader

from unet.unet import *
from unet.preprocess import *
from unet.mydataset import *
from unet.myconfig import *


class Processor:
    def __init__(self, data_path='', batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs

        # GPU/CPU
        self.device = torch.device(DEVICE)

        # 模型、优化器、损失函数
        self.model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        self.loss_fn = torch.nn.BCELoss()  # 二分类交叉熵损失函数

        # 未输入路径，不进行训练
        if data_path == '':
            return

        # 数据集
        self.data = MyDataset(os.path.join(self.data_path, IMG_PATH_NAME), os.path.join(self.data_path, LABEL_PATH_NAME))
        self.data_size = len(self.data)
        self.train_size = int(self.data_size * TRAIN_SIZE)
        self.eval_size = self.data_size - self.train_size
        print(f"train size:{self.train_size}, eval size:{self.eval_size}")

    def _postprocess(self, img):
        """
        后处理：闭运算、均值滤波
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.blur(img, BLUR_KERNEL_SIZE, borderType=cv2.BORDER_REPLICATE)
        img = cv2.threshold(img, POST_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        return img

    def _shuffle(self):
        """
        生成打乱的数据集
        """
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def _train(self, img, label):
        """
        训练
        """
        self.optimizer.zero_grad()
        img = img.to(self.device)
        label = label.to(self.device)
        # 前向传播
        output = self.model(img)
        # 计算损失
        loss = self.loss_fn(output, label)
        print(f"train loss:{loss.item()}")
        # 反向传播
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval(self, img, label):
        """
        验证
        """
        img = img.to(self.device)
        label = label.to(self.device)
        # 前向传播
        output = self.model(img)
        # 计算损失
        loss = self.loss_fn(output, label)
        print(f"eval loss:{loss.item()}")
        return loss.item()

    def _train_and_eval(self, epoch):
        """
        训练与验证
        """
        train_loss = 0
        eval_loss = 0
        num = 0
        print(f"start epoch:{epoch}")

        self.model.train()
        for img, label in self._shuffle():
            # 前6/7作为训练集，后1/7作为验证集
            if num == self.train_size:
                self.model.eval()
            if num < self.train_size:
                train_loss += self._train(img, label)
            else:
                eval_loss += self._eval(img, label)
            num += 1

        print(f"epoch:{epoch}, train loss:{train_loss / self.train_size}, eval loss:{eval_loss / self.eval_size}")
        return eval_loss / self.eval_size

    def run(self):
        """
        开始训练
        """
        for epoch in range(1, self.epochs + 1):
            eval_loss = self._train_and_eval(epoch)
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"./model/thyroid-{epoch}.pth")
                if eval_loss < 0.05:
                    break
            self.scheduler.step()

    def extract(self, model_path):
        """
        加载模型
        :param model_path: 模型路径
        """
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def predict(self, img_path, out_path, threshold=0.5):
        """
        处理图片
        :param img_path: 图片路径
        :param out_path: 导出路径
        :param threshold: mask阈值，默认为0.5
        :return: Image格式甲状腺mask
        """
        img = Image.open(img_path).convert("L")
        img = cropping(img)
        img_scaled = img.resize(IMG_SIZE)
        img_scaled = TF.to_tensor(img_scaled).unsqueeze(0)
        img_scaled = img_scaled.to(self.device)
        self.model.eval()
        output = self.model(img_scaled).squeeze(0)
        output = torch.where(output > threshold, torch.ones_like(output), torch.zeros_like(output))
        out_img = TF.to_pil_image(output).resize(img.size)

        # 将out_img转为原图大小，并进行边缘平滑
        out_img = np.array(out_img)
        img = np.array(img)
        out_img = self._postprocess(out_img)
        out_img = cv2.bitwise_and(img, out_img)
        cv2.imwrite(out_path, out_img)
