import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import time
import datetime

from internal.uniformity import *
from seg_roi import find_roi
from unet.processor import *
from internal.cystic_solid import *
from halo.halo_detector import *
from pos import *
from OCR import * 
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np
from calCa import calcification
from unet.preprocess import cropping
from rec_edge import *
from angle_detect import *

foreGrounddir = '/fore'
roiDir = '/roi'

foreLabelGrounddir = '/fore/label'
roiLabeldir = '/roi/label'

tirads = ['<2%', '2%-10%', '10%-50%', '50%-90%', '50%-90%', '>90%']

# 将单通道的图片存成三通道的图片
def save_rgb(img, save_path):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, rgb)


def bit_wise(src,mask):
    return cv2.bitwise_and(src,src,mask=mask)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.width = 1500
        self.height = 840

        self.setWindowTitle("甲状腺超声诊断")
        self.setFixedSize(self.width, self.height)

        # c-tirads评分
        self.c_tirads = 0

        '''
        参数说明
        img_selected: 选中的图片的路径
        img_seg_roi: 选中的图片经过分割和ROI提取后的图片的路径
        out_roi_dirname: 保存ROI提取后的图片的路径
        需要存储的照片，有原图和分割后的图片
        '''
        self.seg = ""
        self.roi = ""
        self.roi_label = ""
        self.roi_dilate = ""

        self.img_selected = ""
        self.filename = ""
        self.out_dirname = "./outputs"


# 结节的相关参数
############################################################################
        self.markpoint = []
        self.size = []
        self.haveOCR = False

        self.a = 0
        self.b = 0
        self.ifMark = False
        self.edgeInfo = ""
        self.haveCorner = False
###########################################################################         

        # src_except_size: 除去尺寸信息的原图
        self.src_except_size = ""

        # 前景分割
        # self.unet = Processor()
        # self.unet.extract('./unet/model/thyroid-50.pth')
        # 囊实性检测
        self.nodule_grayscale = 0.0
        self.cystic = CysticSolidDetector()
        # 均匀性检测
        self.uniformity = UniformityDetector()
        # 声晕检测
        self.halo = HaloDetector()

        self.ThyroidSeg = nnUNetPredictor(
                        tile_step_size=0.5,
                        use_gaussian=True,
                        use_mirroring=True,
                        perform_everything_on_gpu=True,
                        device=torch.device('cuda', 0),
                        verbose=False,
                        verbose_preprocessing=False,
                        allow_tqdm=True
                    )
        self.ThyroidSeg.initialize_from_trained_model_folder(
                    join(nnUNet_results, 'Dataset119_ThyforegroundSegmentation\\nnUNetTrainer__nnUNetPlans__2d'),
                    use_folds=("all",),
                    checkpoint_name='checkpoint_best.pth',)

        ### nnunet的导入
        self.nnunet_predictor = nnUNetPredictor(
                        tile_step_size=0.5,
                        use_gaussian=True,
                        use_mirroring=True,
                        perform_everything_on_gpu=True,
                        device=torch.device('cuda', 0),
                        verbose=False,
                        verbose_preprocessing=False,
                        allow_tqdm=True
                    )
    
        self.nnunet_predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset120_ThyroidSegmentation\\nnUNetTrainer__nnUNetPlans__2d'),
            use_folds=("all",),
            checkpoint_name='checkpoint_final.pth',
        )

        '''
        报告内容
        '''
        self.orient_report = ""
        self.cystic_report = ""
        self.uniformity_report = ""
        self.halo_report = ""
        self.calci_report = ""

        if not os.path.exists(self.out_dirname):
            os.mkdir(self.out_dirname)
        if not os.path.exists(self.out_dirname + foreGrounddir):
            os.mkdir(self.out_dirname + foreGrounddir)
        if not os.path.exists(self.out_dirname + roiDir):
            os.mkdir(self.out_dirname + roiDir)
        if not os.path.exists(self.out_dirname + foreLabelGrounddir):
            os.mkdir(self.out_dirname + foreLabelGrounddir)
        if not os.path.exists(self.out_dirname + roiLabeldir):
            os.mkdir(self.out_dirname + roiLabeldir)

        # 创建菜单栏
        self.menu = self.menuBar()
        # 1. 文件
        self.file_menu = self.menu.addMenu("文件")

        # 1.1 导入图片
        self.load_file_action = QAction("导入图片", self)
        self.load_file_action.setShortcut("Ctrl+O")
        self.load_file_action.triggered.connect(self.slot_load_file)
        self.file_menu.addAction(self.load_file_action)

        # 1.2 导入文件夹
        self.load_directory_action = QAction("导入文件夹", self)
        self.load_directory_action.setShortcut("Ctrl+D")
        self.load_directory_action.triggered.connect(self.slot_load_directory)
        self.file_menu.addAction(self.load_directory_action)

        # 1.3 保存路径
        self.output_directory_action = QAction("选择保存路径", self)
        self.output_directory_action.setShortcut("Ctrl+S")
        self.output_directory_action.triggered.connect(self.slot_output_directory)
        self.file_menu.addAction(self.output_directory_action)

        # 2. 帮助
        self.help_menu = self.menu.addMenu("帮助")
        self.help_action = QAction("操作说明", self)
        self.help_action.setShortcut("Ctrl+H")
        self.help_action.triggered.connect(self.slot_help)
        self.help_menu.addAction(self.help_action)

        # 生成布局
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 左侧文件列表：上方为原图片，下方为处理后的图片
        self.list_widget = QWidget()
        self.list_widget.setMinimumWidth(200)
        self.list_layout = QVBoxLayout()
        self.list_widget.setLayout(self.list_layout)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(lambda item: self.slot_show_image(item, True))
        self.output_list_widget = QListWidget()
        self.output_list_widget.itemClicked.connect(lambda item: self.slot_show_image(item, False))
        self.list_layout.addWidget(self.file_list_widget)
        self.list_layout.addWidget(self.output_list_widget)
        self.main_layout.addWidget(self.list_widget)

        # 中间显示图片
        self.image_label = QLabel()
        self.image_label.setFixedSize(1024, 768)
        self.main_layout.addWidget(self.image_label)

        # 右侧控制面板：上方为按钮，下方为输出框
        self.control_widget = QWidget()
        self.control_widget.setMinimumWidth(160)
        self.control_layout = QVBoxLayout()
        self.control_layout.setAlignment(Qt.AlignTop)
        self.control_widget.setLayout(self.control_layout)
        self.main_layout.addWidget(self.control_widget)

        # 控制面板的按钮
        self.button_seg = QPushButton("甲状腺分割")
        self.button_seg.clicked.connect(self.slot_button_seg_clicked)
        self.control_layout.addWidget(self.button_seg)

        self.button_nodule = QPushButton("结节分割")
        self.button_nodule.clicked.connect(self.slot_button_nodule_clicked)
        self.control_layout.addWidget(self.button_nodule)

        self.button_orient = QPushButton("探头方向检测")
        self.button_orient.clicked.connect(self.slot_button_orient_clicked)
        self.control_layout.addWidget(self.button_orient)

        self.button_cystic = QPushButton("囊实性检测")
        self.button_cystic.clicked.connect(self.slot_button_cystic_clicked)
        self.control_layout.addWidget(self.button_cystic)

        self.button_uniform = QPushButton("均匀性检测")
        self.button_uniform.clicked.connect(self.slot_button_uniform_clicked)
        self.control_layout.addWidget(self.button_uniform)

        self.button_halo = QPushButton("声晕检测")
        self.button_halo.clicked.connect(self.slot_button_halo_clicked)
        self.control_layout.addWidget(self.button_halo)

        self.button_calci = QPushButton("钙化检测")
        self.button_calci.clicked.connect(self.slot_button_calci_clicked)
        self.control_layout.addWidget(self.button_calci)

        self.button_generate = QPushButton("生成报告")
        self.button_generate.clicked.connect(self.slot_button_generate_clicked)
        self.control_layout.addWidget(self.button_generate)

        self.button_clear = QPushButton("清空工作区")
        self.button_clear.clicked.connect(self.slot_button_clear_clicked)
        self.control_layout.addWidget(self.button_clear)

        # 控制面板的输出框
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.control_layout.addWidget(self.text_edit)

    def log_message(self, message, *args):
        """
        输出日志：格式为：[时间] message
        """
        msg = message % args
        msg = "[%s] %s\n\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg)
        # 将msg加到开头
        self.text_edit.setText(msg + self.text_edit.toPlainText())

    def slot_load_file(self):
        file_names, file_type = QFileDialog.getOpenFileNames(self, "打开文件", "./unet/dataset/images-ori",
                                                             "Image Files(*.png *.jpg *.bmp)")
        if len(file_names) == 0:
            return

        for file_name in file_names:
            print("导入图片：", file_name)
            new_item = QListWidgetItem(file_name)
            self.file_list_widget.addItem(new_item)
            self.slot_show_image(new_item, True)

    def slot_load_directory(self):
        dirname = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if dirname == "":
            return

        print("导入文件夹：", dirname)
        files = os.listdir(dirname)
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".bmp"):
                new_item = QListWidgetItem(os.path.join(dirname, file))
                self.file_list_widget.addItem(new_item)

        if len(files) > 0:
            self.slot_show_image(self.file_list_widget.item(self.file_list_widget.count() - 1), True)

    def slot_output_directory(self):
        dirname = QFileDialog.getExistingDirectory(self, "选择保存路径", "./")
        if dirname == "":
            return

        print("保存路径：", dirname)
        self.out_dirname = dirname

    def slot_show_image(self, item, select):
        print("显示图片：", item.text())
        self.image_label.setPixmap(QPixmap(item.text()))
        # 导入的图片一定是src，只能导入这个图片，存储的是路径
        # 刚导入图片的时候，还没有进行分割，分割的那些东西都是空的
        if select:
            self.img_selected = item.text()
            self.filename = os.path.basename(self.img_selected)
            self.src_except_size = os.path.join(self.out_dirname, self.filename)
            old_image = cv2.imread(self.img_selected, cv2.IMREAD_GRAYSCALE)
            new_image = cropping(old_image)
            cv2.imwrite(self.src_except_size, new_image)
            self.reset()
            self.haveOCR ,self.size = get_size_interface(item.text())
            if(self.haveOCR == True):
                self.log_message("甲状腺结节的大小为%s", self.size)
                self.a = self.size[0]
                self.b = self.size[1]
            else:
                self.log_message("未检测到甲状腺结节大小OCR,使用像素点进行匹配，每厘米对应像素点为%d", self.size)
        else:
            self.img_selected = ""
            self.reset()

    def slot_button_seg_clicked(self):
        """
        点击事件：开始前景分割
        """
        if self.src_except_size == "":
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        self.seg_processing()

    def slot_button_nodule_clicked(self):
        """
        点击事件：开始结节分割
        """
        if self.seg == "":
            QMessageBox.information(self, "提示", "请先进行甲状腺前景分割", QMessageBox.Ok)
            return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        self.nodule_processing()

    def slot_button_orient_clicked(self):
        """
        点击事件：开始探头方向检测
        """
        if self.img_selected == "":
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        self.orient_processing()

    def slot_button_cystic_clicked(self):
        """
        点击事件：开始囊实性检测
        """
        if self.seg == "":
            QMessageBox.information(self, "提示", "请先进行前景分割", QMessageBox.Ok)
            return
        if self.roi == "":
            QMessageBox.information(self, "提示", "请先进行结节roi分割", QMessageBox.Ok)
            return

        self.cystic_processing()

    def slot_button_uniform_clicked(self):
        """
        点击事件：开始均匀性检测
        """
        if self.roi == "":
            QMessageBox.information(self, "提示", "请先进行结节roi分割", QMessageBox.Ok)
            return

        self.uniform_processing()

    def slot_button_halo_clicked(self):
        """
        点击事件：开始声晕检测
        """
        if self.roi == "":
            QMessageBox.information(self, "提示", "请先进行结节roi分割", QMessageBox.Ok)
            return

        self.halo_processing()

    def slot_button_calci_clicked(self):
        """
        点击事件：开始钙化检测
        """
        if self.img_selected == "":
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return
        if self.out_dirname == "":
            QMessageBox.information(self, "提示", "请先选择保存路径", QMessageBox.Ok)
            return

        self.calci_processing()

    def slot_button_generate_clicked(self):
        """
        点击事件：生成报告
        """
        # 用户输入报告名称
        report_name, ok = QInputDialog.getText(self, "输入报告名称", "请输入报告名称：")
        if not ok:
            return
        if report_name == "":
            QMessageBox.information(self, "提示", "请输入正确的报告名称", QMessageBox.Ok)
            return
        if not report_name.endswith(".html"):
            report_name += ".html"

        if not self.size:
            QMessageBox.information(self, "提示", "请先选择图片", QMessageBox.Ok)
            return
        if self.seg == "":
            QMessageBox.information(self, "提示", "请先进行前景分割", QMessageBox.Ok)
            return
        if self.roi == "":
            QMessageBox.information(self, "提示", "请先进行结节roi分割", QMessageBox.Ok)
            return
        if self.orient_report == "":
            QMessageBox.information(self, "提示", "请先进行探头方向检测", QMessageBox.Ok)
            return
        if self.cystic_report == "":
            QMessageBox.information(self, "提示", "请先进行囊实性检测", QMessageBox.Ok)
            return
        if self.uniformity_report == "":
            QMessageBox.information(self, "提示", "请先进行均匀性检测", QMessageBox.Ok)
            return

        self.output_report(report_name)

    def slot_button_clear_clicked(self):
        """
        点击事件：清空
        """
        self.file_list_widget.clear()
        self.output_list_widget.clear()
        self.image_label.clear()
        self.text_edit.clear()
        self.img_selected = ""
        self.out_dirname = "./outputs"
        self.reset()

    def slot_help(self):
        QMessageBox.information(self, "操作说明", "选择一张超声图后，请依次点击右侧控制台的按钮\n", QMessageBox.Ok)

    def seg_processing(self):
        """
        前景分割
        """
        print("开始分割计算：", self.src_except_size)
        out_path = self.out_dirname + foreLabelGrounddir + '/' + self.filename
        out_path = out_path.replace(".png", "")
        # self.unet.predict(self.img_selected, out_path)
        self.nnunet_seg_Thyroid(out_path)
        out_path = out_path + ".png"
        self.log_message("保存路径：%s", out_path)
        
        tmp_thyroid_label = cv2.imread(out_path)
        tmp_src = cv2.imread(self.src_except_size)
        tmp_seg = (tmp_src * tmp_thyroid_label)
        tmp_seg = cv2.cvtColor(tmp_seg, cv2.COLOR_BGR2RGB)

        self.seg = self.out_dirname + foreGrounddir + '/' + self.filename
        cv2.imwrite(self.seg, tmp_seg)

        new_item = QListWidgetItem(self.seg)
        self.output_list_widget.addItem(new_item)
        self.image_label.setPixmap(QPixmap(self.seg))
        print("分割计算结束：", out_path)

    def nodule_processing(self):
        """
        结节分割
        """
        # self.img_selected: 输入图片路径
        # self.log_message(self, message): 右侧输出框输出信息
        print("开始分割roi:", self.seg)
        out_path = self.out_dirname + roiLabeldir + '/' + self.filename
        out_path = out_path.replace(".png", "")
        self.nnunet_seg_roi(out_path)
        out_path = out_path + ".png"
        self.roi_label = out_path
        self.log_message("保存路径：%s", self.roi_label)

        tmp_seg = cv2.imread(self.seg)
        tmp_roi_label = cv2.imread(out_path)
        tmp_roi = (tmp_seg * tmp_roi_label)

        self.roi = self.out_dirname + roiDir + '/' + self.filename
        cv2.imwrite(self.roi, tmp_roi)

        self.ifMark, self.markpoint = find_roi(self.src_except_size, self.seg)
        print("markpoint:", self.markpoint)
        new_item = QListWidgetItem(self.roi)
        self.output_list_widget.addItem(new_item)
        self.image_label.setPixmap(QPixmap(self.roi))
        print("分割roi结束：", out_path)
        self.edgeInfo, tmp_score = analyze_edge(self.seg,self.roi)

        self.haveCorner = angle_detect(self.roi_label)
        if tmp_score > 0 or self.haveCorner:
            self.c_tirads += 1
        self.log_message("%s, 形状%s", self.edgeInfo, "不规则" if self.haveCorner else "规则")
        
        if not self.haveOCR:
            scale = self.size
            self.a, self.b = get_frame(self.roi_label)
            self.a = self.a/scale
            self.b = self.b/scale
            self.size = [self.a,self.b]
            self.log_message("a:%s, b:%s", self.a, self.b)

    def orient_processing(self):
        """
        探头方向检测
        """
        # self.img_selected: 输入图片路径
        # self.log_message(self, message): 右侧输出框输出信息
        # 切面方向orient = 1（横切且左叶）2（纵切）3（横切且右叶）0（无法判断）
        result,orient = get_icon(self.img_selected,self.seg)
        result+="\n"
        desc, nodule_orient = check_roi_pos(self.seg,self.roi_label,orient,self.a,self.b,self.haveOCR)
        result+=desc
        # print(result)
        self.orient_report = result
        self.log_message("%s", self.orient_report)
        if nodule_orient > 0:
            self.c_tirads += 1

    def cystic_processing(self):
        """
        囊实性检测
        """
        print("开始囊实性检测计算：%s, %s" % (self.seg, self.roi))
        self.cystic_report, score, self.nodule_grayscale = self.cystic.cystic_solid_detect(self.seg, self.roi)
        self.log_message("%s, %f", self.cystic_report, self.nodule_grayscale)
        if score >= 0.9:
            self.c_tirads += 1

    def uniform_processing(self):
        """
        均匀性检测
        """
        print("开始均匀性检测：%s" % self.roi)
        self.uniformity_report, score = self.uniformity.uniformity_detect(self.roi)
        self.log_message("%s, %f", self.uniformity_report, score)
        if score >= 0.3:
            self.u_tirads += 1

    def halo_processing(self):
        """
        声晕检测
        """
        print("开始声晕检测：%s" % self.roi)
        if len(self.markpoint) < 4:
            self.log_message("暂不支持打点不完整结节的声晕检测")
            return

        img = cv2.imread(self.roi_label)
        img = cv2.dilate(img, np.ones((15, 15), np.uint8), iterations=1)
        img_ori = cv2.imread(self.src_except_size)
        img = (img * img_ori)
        self.roi_dilate = self.out_dirname + '/roi_dilate_' + self.filename
        cv2.imwrite(self.roi_dilate, img)

        # 找到最左侧的点
        tmp = 1e5
        p1 = [0, 0]
        for i in range(4):
            if self.markpoint[i][0] < tmp:
                tmp = self.markpoint[i][0]
                p1 = self.markpoint[i]

        # 找到最右侧的点
        tmp = -1e5
        p2 = [0, 0]
        for i in range(4):
            if self.markpoint[i][0] > tmp:
                tmp = self.markpoint[i][0]
                p2 = self.markpoint[i]

        # 找到最上侧的点
        tmp = 1e5
        p3 = [0, 0]
        for i in range(4):
            if self.markpoint[i][1] < tmp:
                tmp = self.markpoint[i][1]
                p3 = self.markpoint[i]

        # 找到最下侧的点
        tmp = -1e5
        p4 = [0, 0]
        for i in range(4):
            if self.markpoint[i][1] > tmp:
                tmp = self.markpoint[i][1]
                p4 = self.markpoint[i]

        hasHalo, isComplete, isEven, roi = self.halo.halo_detect(self.roi_dilate, p1, p2, p3, p4)
        if hasHalo:
            self.halo_report = "声晕%s, 宽度%s" % ("完整" if isComplete else "不完整", "均匀" if isEven else "不均匀")
        else:
            self.halo_report = "未检测到声晕"
        self.log_message(self.halo_report)

        name = self.out_dirname + "/halo_" + self.filename
        cv2.imwrite(name, roi)
        new_item = QListWidgetItem(name)
        self.output_list_widget.addItem(new_item)
        self.image_label.setPixmap(QPixmap(name))

    def calci_processing(self):
        """
        钙化检测
        """
        # self.img_selected: 输入图片路径
        # self.log_message(self, message): 右侧输出框输出信息
        # 需要哪些额外参数或接口，请联系tcy
        self.calci_report, roi, count1, count2 = calcification(self.a,self.b,self.roi,self.roi_label,self.nodule_grayscale,self.markpoint)
        self.log_message(self.calci_report)
        if count2 > 0:
            self.c_tirads += 1

        name = self.out_dirname + "/calci_" + self.filename
        cv2.imwrite(name, roi)
        new_item = QListWidgetItem(name)
        self.output_list_widget.addItem(new_item)
        self.image_label.setPixmap(QPixmap(name))
    
    # 分割前景
    def nnunet_seg_Thyroid(self,outpath):
        self.ThyroidSeg.predict_from_files([[self.src_except_size]],
                                 [outpath],
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        
    # 分割ROI
    def nnunet_seg_roi(self,outpath):
        self.nnunet_predictor.predict_from_files([[self.seg]],
                                 [outpath],
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # 每当传入一张src的时候，需要讲之前的信息重新修改掉
    def reset(self):
        self.seg = ""
        self.roi = ""
        self.markpoint = []
        self.c_tirads = 0
        self.size = []
        self.orient_report = ""
        self.cystic_report = ""
        self.uniformity_report = ""
        self.halo_report = ""

    # 生成报告
    def output_report(self, name):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        possibility = tirads[self.c_tirads]
        report_content = f"""
        <h3>
            超声描述：
        </h3>
        <p>
            甲状腺结节形态信息：
            结节大小为{self.size[0]}cm*{self.size[1]}cm;
            {self.cystic_report};
            {self.uniformity_report};
            {self.edgeInfo},
            {"形状不规则" if self.haveCorner else "形状规则"};
            {self.calci_report}{"" if self.halo_report == "" else ";"}
            {self.halo_report}.
        </p>
        <p>
            甲状腺位置信息：{self.orient_report}
        </p>
        <p>
            甲状腺结节TI-RADS等级：{self.c_tirads}
            恶性概率：{possibility}
        </p>
        <span style="color:red">
            <p>
                请结合临床综合分析。
            </p>
        </span>
        <p>
            {today}
        </p>
        """
        with open(self.out_dirname + '/' + name, "w") as f:
            f.write(report_content)

        self.log_message("报告保存位置: %s/%s", self.out_dirname, name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
