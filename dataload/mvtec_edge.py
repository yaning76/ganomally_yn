# -*- coding:utf8 -*-
# @TIME     : 2020/12/7 10:12
# @Author   : SuHao# -*- coding:utf8 -*-
# @TIME     : 2020/12/4 15:48
# @Author   : SuHao
# @File     : mvtec.py

'''
reference: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
'''


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import cv2 
import skimage
from skimage.feature import local_binary_pattern
# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES =['MainboardScrew05','MainboardScrew04','MainboardScrew02','MainboardScrew01','MainboardScrew1',
              'MainboardScrew','MainboardScrew_','CPUScrew','bottle', 'cable', 
               'capsule', 'carpet', 'grid','grid_',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset_edge(Dataset):
    def __init__(self, dataset_path='../data/', class_name='bottle', is_train=True, resize=128):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()



    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # set transforms
        angle = np.random.randint(0, 360)
        self.transform_x = T.Compose([T.Resize(self.resize, Image.ANTIALIAS),
                                      T.ToTensor(),])
                                      # T.Normalize(mean=(0.5, 0.5, 0.5),
                                      #             std=(0.5, 0.5, 0.5))])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.ToTensor(),])
#         x = Image.open(x).convert('RGB')
        x = Image.fromarray(np.uint8(x))
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
#         print('img_dir,gt_dir',img_dir,gt_dir)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if (f.endswith('.png') or f.endswith('.jpg'))])
            for f in os.listdir(img_type_dir):
                image_path=os.path.join(img_type_dir, f)
#                     aug = self.randAugmenter()
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                s = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                e = cv2.erode(gray, s)
                d = cv2.dilate(gray, s)
                img_new=d-e
                img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
                x.append(img_new)
            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))#标签为0
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))#标签为1
                gt_type_dir = os.path.join(gt_dir, img_type)
#                 print('gt_type_dir',gt_type_dir)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
#                 if self.class_name=='MainboardScrew02'or self.class_name=='MainboardScrew05':
#                     gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.jpg')
#                                  for img_fname in img_fname_list]
#                 else:
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                             for img_fname in img_fname_list]
#                 gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.jpg')
#                                  for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

# @File     : dataload.py

import torchvision.transforms as T
import torchvision.datasets as dset
from dataload.mvtec import MVTecDataset
def load_dataset(dataroot, dataset_name, imageSize, trans, train=True):
    params_med = {"dataroot": dataroot, "split": 'train' if train else 'test', "transform":trans}
    if dataset_name == "mnist":
        dataset = dset.MNIST(root=dataroot,
                             train=train,
                             download=True,
                             transform=T.Compose([T.Resize(imageSize), T.ToTensor()]),
                             )
    else:
        dataset = MVTecDataset(dataroot, class_name=dataset_name, is_train=train, resize=imageSize)
    return dataset