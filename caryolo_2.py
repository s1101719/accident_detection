import sys
import cv2
import argparse
import random
import torch
import torchvision
import timm
import numpy as np
import yaml
import torch.backends.cudnn as cudnn
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import shutil
import easyocr

from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from contextlib import contextmanager
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize,InterpolationMode
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from torchvision import transforms
from PIL import Image
from ReID.load_model import load_model_from_opts
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
 
result_text=[]
#ReID definition
def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.detected_objects = []  # List to store detected objects

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix
    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512,
                 model_subtype="50", mixstyle=True):
        super(ft_net, self).__init__()
        if model_subtype in ("50", "default"):
            if ibn:
                model_ft = torch.hub.load(
                    'XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            else:
                model_ft = models.resnet50(weights="IMAGENET1K_V2")
        elif model_subtype == "101":
            if ibn:
                model_ft = torch.hub.load("XingangPan/IBN-Net", "resnet101_ibn_a", pretrained=True)
            else:
                model_ft = models.resnet101(weights="IMAGENET1K_V2")
        elif model_subtype == "152":
            if ibn:
                raise ValueError("Resnet152 has no IBN variants available.")
            model_ft = models.resnet152(weights="IMAGENET1K_V2")
        else:
            raise ValueError(f"Resnet model subtype: {model_subtype} is invalid, choose from: ['50','101','152'].")

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=circle)
        self.mixstyle = MixStyle(alpha=0.3) if mixstyle else None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer2(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


def load_weights(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if model.classifier.classifier[0].weight.shape != state["classifier.classifier.0.weight"].shape:
        state["classifier.classifier.0.weight"] = model.classifier.classifier[0].weight
        state["classifier.classifier.0.bias"] = model.classifier.classifier[0].bias
    model.load_state_dict(state)
    return model


def create_model(n_classes, kind="resnet", **kwargs):
    """Creates a model of a given kind and number of classes"""
    if kind == "resnet":
        return ft_net(n_classes, **kwargs)
    else:
        raise ValueError("Model type cannot be created: {}".format(kind))


def load_model(n_classes, kind="resnet", ckpt=None, remove_classifier=False, **kwargs):
    model = create_model(n_classes, kind, **kwargs)
    if ckpt:
        model = load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model


def load_model_from_opts(opts_file, ckpt=None, return_feature=False, remove_classifier=False):

    with open(opts_file, "r") as stream:
        opts = yaml.load(stream, Loader=yaml.FullLoader)
    n_classes = opts["nclasses"]
    droprate = opts["droprate"]
    stride = opts["stride"]
    linear_num = opts["linear_num"]

    model_subtype = opts.get("model_subtype", "default")
    model_type = opts.get("model", "resnet_ibn")
    mixstyle = opts.get("mixstyle", False)

    if model_type in ("resnet", "resnet_ibn"):
        model = create_model(n_classes, "resnet", droprate=droprate, ibn=(model_type == "resnet_ibn"),
                             stride=stride, circle=return_feature, linear_num=linear_num,
                             model_subtype=model_subtype, mixstyle=mixstyle)
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    if ckpt:
        load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model

def fliplr(img):
    #"""flip images horizontally in a batch"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, X, device="cuda"):
    #"""Exract the embeddings of a single image tensor X"""
    if len(X.shape) == 3:
        X = torch.unsqueeze(X, 0)
    X = X.to(device)
    feature = model(X).reshape(-1)

    X = fliplr(X)
    flipped_feature = model(X).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)


def get_scores(query_feature, gallery_features):
    #"""Calculate the similarity scores of the query and gallery features"""
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_features, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.timer_video1 = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
 
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='./weights/accident_detection.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='data/images', help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.80, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.9, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        #print(self.opt)
 
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
 
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
 
        cudnn.benchmark = True
 
        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
 
        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
 
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 1600)
        font = QtGui.QFont()
        font.setPointSize(20)  # 設定字體大小為16
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 260, 224, 68))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFont(font)
        #self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        #self.pushButton_2.setGeometry(QtCore.QRect(40, 440, 224, 68))
        #self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(40, 440, 224, 68))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setFont(font)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(280, 180, 1400, 600))
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setFont(font)
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setGeometry(QtCore.QRect(280, 800, 700, 500))
        self.groupBox1.setObjectName("groupBox1")
        self.groupBox1.setFont(font)
        self.groupBox2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox2.setGeometry(QtCore.QRect(1000, 800, 650, 500))
        self.groupBox2.setObjectName("groupBox2")
        self.groupBox2.setFont(font)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 40, 1100, 725))
        self.label.setObjectName("label")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(520, 1000, 250, 42))
        self.label1.setObjectName("label1")
        self.label1.setFont(font)
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(660, 1000, 280, 42))
        self.label2.setObjectName("label2")
        self.label2.setFont(font)
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(520, 1060, 500, 42))
        self.label3.setObjectName("label3")
        self.label3.setFont(font)
        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setGeometry(QtCore.QRect(705, 1000, 142, 42))
        self.label4.setObjectName("label4")
        self.label4.setFont(font)
        self.label5 = QtWidgets.QLabel(self.centralwidget)
        self.label5.setGeometry(QtCore.QRect(520, 1200, 142, 500))
        self.label5.setObjectName("label5")
        self.label5.setFont(font)
        self.label6 = QtWidgets.QLabel(self.groupBox1)
        self.label6.setGeometry(QtCore.QRect(10, 30, 600, 100))
        self.label6.setObjectName("label6")
        self.label6.setFont(font)
        self.label7 = QtWidgets.QLabel(self.groupBox2)
        self.label7.setGeometry(QtCore.QRect(10, 10, 700, 500))
        self.label7.setObjectName("label7")
        self.label7.setFont(font)
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self) 
        self.media_player.setVideoOutput(self.video_widget) 
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(300, 22, 942, 102))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 60))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    #ReID
    def ReID(self,output_path):
        folder_name = "ReID_images"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        #filelist='caryoloimages'
        #for file_name in filelist:
        
        interpolate = torchvision.transforms.functional.InterpolationMode.BICUBIC
        data_transforms = transforms.Compose([
                transforms.Resize((224, 224), interpolation=interpolate),
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB from grayscale
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load the vehicle detection and recognition model
        model = load_model_from_opts("./ReID/model/result/opts.yaml", ckpt="./ReID/model/result/net_18.pth", remove_classifier=True)
        model.eval()
        model.to("cuda")
        # Read the query image
        #query_image = Image.open(os.path.join('caryoloimages', file_name))
        query_image = Image.open(output_path)
        query_image = query_image.resize((224, 224))  # Properly resize the image
        X_query = torch.unsqueeze(data_transforms(query_image), 0).to("cuda")

        # Load and preprocess gallery images
        upload_path = "./ReID/test"
        upload_images = os.listdir(upload_path)
        gallery_images = []
        for img_path in upload_images:
                img = Image.open(os.path.join(upload_path, img_path))
                img = img.resize((224, 224))  # Properly resize the image
                gallery_images.append(img)

        X_gallery = torch.stack(tuple(map(data_transforms, gallery_images))).to("cuda")

        # Compute features for query and gallery images
        f_query = extract_feature(model, X_query).detach().cpu()
        f_gallery = [extract_feature(model, X) for X in X_gallery]
        f_gallery = torch.stack(f_gallery).detach().cpu()

        # Compute similarity scores
        scores = get_scores(f_query, f_gallery)
        # Find the index of the image with the highest similarity score
        max_score_index = scores.argmax().item()
        reid_path=os.path.join(upload_path, upload_images[max_score_index])
        showimg=cv2.imread(reid_path)
        cv2.imwrite(os.path.join(folder_name, upload_images[max_score_index]),showimg)
        return reid_path
    
    def load_LPDetection_weights(self,reid_path):
        folder_name = "LP_images"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.model = attempt_load("./weights/license_plate_detection.pt", map_location=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=stride) 
        if self.half:
            self.model.half() 
            #print(file_name)
            #reid_image = cv2.imread(os.path.join('reid_images', file_name)) 
            reid_image = cv2.imread(reid_path) 
            showimg=reid_image
            with torch.no_grad():
                reid_image = letterbox(reid_image, new_shape=self.imgsz)[0]
                reid_image = reid_image[:, :, ::-1].transpose(2, 0, 1)
                reid_image = np.ascontiguousarray(reid_image)
                reid_image = torch.from_numpy(reid_image).to(self.device)
                reid_image = reid_image.half() if self.half else reid_image.float()  # uint8 to fp16/32
                reid_image /= 255.0  # 0 - 255 to 0.0 - 1.0
                if reid_image.ndimension() == 3:
                    reid_image = reid_image.unsqueeze(0)
                pred = self.model(reid_image, augment=False)[0]
                pred = non_max_suppression(pred, 0.85, 0.99, classes=None,
                                        agnostic=False)
                num=0
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(reid_image.shape[2:], det[:, :4], showimg.shape).round()
                        for *xyxy, conf in reversed(det):
                            cropped_object = showimg[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            output_path = os.path.join('LP_images', 'LP_'f'{num}.jpg')
                            num=num+1
                            cv2.imwrite(output_path, cropped_object)
                        reader = easyocr.Reader(['en']) 
                        LP_result = reader.readtext(output_path)
                        #result_text.append(LP_result)
                #print(result_text)
        return LP_result
        #result_text_str=""
        #for text in result_text:
        #    result_text_str=result_text_str+str(text[0][1])+" "
        #print(result_text_str)
        #_translate = QtCore.QCoreApplication.translate
        #self.label3.setText(_translate("MainWindow", "車牌：" + str(result_text_str)))   


    def crop_and_save_objects(self, img, detections, names, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for det in detections:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    cls_name = names[int(cls)]
                    cropped_object = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    output_path = os.path.join(output_folder, f'{cls_name}_{conf:.2f}.jpg')
                    cv2.imwrite(output_path, cropped_object)
        return output_path

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ShieldMyRide 車禍即時通報系統"))
        self.pushButton.setText(_translate("MainWindow", "圖片檢測"))
        #self.pushButton_2.setText(_translate("MainWindow", "攝像頭檢測"))
        self.pushButton_3.setText(_translate("MainWindow", "影片檢測"))
        self.groupBox.setTitle(_translate("MainWindow", "車禍檢測结果"))
        self.groupBox1.setTitle(_translate("MainWindow", "警方收到的訊息"))
        self.groupBox2.setTitle(_translate("MainWindow", "緊急聯絡人收到的訊息"))
        #self.label3.setText(_translate("MainWindow", "車牌："))
        #self.label1.setText(_translate("MainWindow", "監視器編號："))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.textEdit.setHtml(_translate("MainWindow",
            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
            "p, li { white-space: pre-wrap; }\n"
            "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
            "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">車輛即時通報演示系统</span></p></body></html>"))
 
    def init_slots(self):
        self.pushButton.clicked.connect(self.button_image_open)
        self.pushButton_3.clicked.connect(self.button_video_open)
        #self.pushButton_2.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)
 
    def init_logo(self):
        pix = QtGui.QPixmap('wechat.jpg')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
 
    def button_image_open(self,output_path):
        _translate = QtCore.QCoreApplication.translate
        #self.label3.setText(_translate("MainWindow", "車牌："))
        #self.label2.setText(_translate("MainWindow", "監視器編號:"))
        folder_path=['caryoloimages','LP_images','ReID_images']
        for folder in folder_path:
            try:
                shutil.rmtree(folder)
                print(f"Delete '{folder} folder successfully' ")
            except:
                print(f"Failed to delete {folder} folder")

        #print('button_image_open')
        name_list = []
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return
 
        img = cv2.imread(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.8, 0.9, classes=False,
                                       agnostic=False)
            #print(pred)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # 呼叫crop_and_save_objects方法進行裁切和儲存
                        output_path=self.crop_and_save_objects(showimg, [det], self.names, 'caryoloimages')
        cv2.imwrite('prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(
            self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        reid_path=self.ReID(output_path)
        self.load_LPDetection_weights(reid_path)
        print("Finish detection")
        return 

    def button_video_open(self):
        self.detected_objects = []  
        folder_path=['caryoloimages','LP_images','ReID_images']
        for folder in folder_path:
            try:
                shutil.rmtree(folder)
                print(f"Delete '{folder} folder successfully' ")
            except:
                print(f"Failed to delete {folder} folder")
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "*.mp4;;*.avi;;All Files(*)")
 
        if not video_name:
            return
 
        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"Failed to open video", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_3.setDisabled(True)
            self.pushButton.setDisabled(True)
            _translate = QtCore.QCoreApplication.translate
            #self.label3.setText(_translate("MainWindow", "車牌："))
            #self.pushButton_2.setDisabled(True)

    def show_video_frame(self):
        name_list = []
        output_path=None
        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]
 
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                            # 呼叫crop_and_save_objects方法進行裁切和儲存
                            output_path=self.crop_and_save_objects(showimg, [det], self.names, 'caryoloimages')
                            if output_path is not None:
                                self.detected_objects.append(output_path)
                            #self.ReID(output_path)
            self.out.write(showimg)  
            show = cv2.resize(showimg, (1200, 550))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            _translate = QtCore.QCoreApplication.translate
            #self.label4.setText(_translate("MainWindow", "237")) 
            LP_result_str=""
            result_text=[]
            for output_path in self.detected_objects:
                reid_path = self.ReID(output_path)
                LP_result=self.load_LPDetection_weights(reid_path)
                if LP_result[0][1] not in result_text:
                    result_text.append(LP_result[0][1])
                    print(LP_result[0][1])
                    LP_result_str=LP_result_str+str(LP_result[0][1])+" "
               
            #self.label3.setText(_translate("MainWindow", "車牌：" + str(LP_result_str))) 
            self.label7.setText(_translate("MainWindow","已傳送訊息給緊急連絡人\n車主：陳OO\n"+"車牌:"+result_text[0]+"\n附近位置:新北勢五股區特二號快速通路\n\n\n"+"車主：王OO\n"+"車牌:"+result_text[0]+"\n附近位置:新北勢五股區特二號快速通路")) 
            self.label6.setText(_translate("MainWindow", "監視器編號 237 發生車禍!\n已傳送影像!")) 
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.video_widget.setGeometry(300, 930, 650, 360)
            self.media_player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile('prediction.avi')))
            self.media_player.play()
            print('Finish Detection')
            self.label.clear()
            self.pushButton_3.setDisabled(False)
            self.pushButton.setDisabled(False)
            #self.pushButton_2.setDisabled(False)
            self.init_logo()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())