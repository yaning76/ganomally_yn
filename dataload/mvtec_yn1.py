# -*- coding:utf8 -*-
# @TIME     : 2020/12/4 15:48
# @Author   : SuHao
# @File     : mvtec.py

'''
reference: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
'''
import random
from math import *
import cv2
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
# import imgaug.augmenters as iaa
from dataload.perlin import rand_perlin_2d_np
# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES =['MainboardScrew05','MainboardScrew04','MainboardScrew02','MainboardScrew01','MainboardScrew1',
              'MainboardScrew','MainboardScrew_','CPUScrew','bottle', 'cable', 
               'capsule', 'carpet', 'grid','grid_',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
def rotate_img(img):
    x= random.randint(0, 360)
    degree=x
    height,width = img.shape[:2]

    M = cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    M[0, 2] += (widthNew - width) / 2  
    M[1, 2] += (heightNew - height) / 2  

    im_rotate = cv2.warpAffine(img,M,(widthNew,heightNew), borderValue=(255, 255, 255))
    return im_rotate
def rotate_image(img1):
    h, w, c = img1.shape
    w_ran=random.randint(1,6)
    h_ran=random.randint(1,6)
    M = cv2.getRotationMatrix2D((w / w_ran, h / h_ran), 45, 1)
    # 执行旋转, 任意角度旋转
    result1 = cv2.warpAffine(img1, M, (w, h))
    return result1
class MVTecDataset(Dataset):
    def __init__(self, dataset_path='../data/', class_name='bottle', is_train=True, resize=128):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.cropsize = cropsize

        # load dataset
#         self.x, self.y, self.mask = self.load_dataset_folder()
        self.x, self.y, self.mask = self.augment_image()
#         self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
#                       iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
#                       iaa.pillike.EnhanceSharpness(),
#                       iaa.AddToHueAndSaturation((-50,50),per_channel=True),
#                       iaa.Solarize(0.5, threshold=(32,128)),
#                       iaa.Posterize(),
#                       iaa.Invert(),
#                       iaa.pillike.Autocontrast(),
#                       iaa.pillike.Equalize(),
#                       iaa.Affine(rotate=(-45, 45))
#                       ]

#         self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])



    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # set transforms
        self.transform_x = T.Compose([T.Resize(self.resize, Image.ANTIALIAS),
                                      T.ToTensor(),])
                                      # T.Normalize(mean=(0.5, 0.5, 0.5),
                                      #             std=(0.5, 0.5, 0.5))])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.ToTensor(),])
        phase = 'train' if self.is_train else 'test'
        if phase == 'train':
            x = Image.fromarray(np.uint8(x))
            x = self.transform_x(x)
            
            mask = Image.open(mask).convert('RGB')
            mask = self.transform_mask(mask)
        else:
            x = Image.fromarray(np.uint8(x))
            x = self.transform_x(x)
            mask = Image.open(mask).convert('RGB')
            mask = self.transform_mask(mask)
#             if y == 0:
#                 mask = torch.zeros([1, self.resize, self.resize])
#             else:
#                 mask = Image.open(mask)
#                 mask = self.transform_mask(mask)

        return x, y, mask
    
    def make_anomaly(self,img):
        img_=[]
        for i in range(4):
            im_rotate=rotate_img(img)
            h,w=im_rotate.shape[:2]
            h1=int(h/2)-64
            h2=int(h/2)+64
            w1=int(w/2)-64
            w2=int(w/2)+64
            img1 = im_rotate[h1:h2, w1:w2]
            img_.append(img1)
        res=np.zeros([256,256,3],np.uint8)
        res[0:128,0:128] = img_[0]
        res[128:256,0:128] = img_[1]
        res[0:128,128:256] = img_[2]
        res[128:256,128:256] = img_[3]
        return res
        
    def augment_image(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)

        if phase == 'train':
            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                for f in os.listdir(img_type_dir):
                    image_path=os.path.join(img_type_dir, f)
#                     aug = self.randAugmenter()
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, dsize=(self.resize, self.resize))
#                     x.append(image)
#                     mask.append(image_path)
#                     mask_image=image
#                     anomaly_source_img=image
                    perlin_scale = 6
                    min_perlin_scale = 0
                    anomaly_img_augmented =self.make_anomaly(image)#原图像增强
                    for i in range(2):
#                         anomaly_img_augmented =aug(image=anomaly_source_img)#原图像增强
                        
                        
                        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
                        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
                        perlin_noise = rand_perlin_2d_np((self.resize, self.resize), (perlin_scalex, perlin_scaley))
#                         perlin_noise = self.rot(image=perlin_noise)#噪声增强
                        threshold = 0.5
                        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                        perlin_thr = np.expand_dims(perlin_thr, axis=2)
#                         print(perlin_thr.shape)
                        
                        anomaly_mask=np.zeros([256,256,3],np.uint8)+perlin_thr

                        img_thr = anomaly_img_augmented.astype(np.float32) * anomaly_mask
                        image_=image.astype(np.float32) * (1-anomaly_mask)
                        augmented_image = image_ +img_thr
                        x.append(augmented_image)
                        mask.append(image_path)
                        
                y.extend([0] * len(x))#标签为0
            y=list(y)
        else:
            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                n=0
                for f in os.listdir(img_type_dir):
                    image_path=os.path.join(img_type_dir, f)
#                     aug = self.randAugmenter()
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, dsize=(self.resize, self.resize))
                    perlin_scale = 6
                    min_perlin_scale = 0
                    anomaly_img_augmented =self.make_anomaly(image)#原图像增强
                    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
                    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
                    perlin_noise = rand_perlin_2d_np((self.resize, self.resize), (perlin_scalex, perlin_scaley))
#                         perlin_noise = self.rot(image=perlin_noise)#噪声增强
                    threshold = 0.5
                    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                    perlin_thr = np.expand_dims(perlin_thr, axis=2)
                    anomaly_mask=np.zeros([256,256,3],np.uint8)+perlin_thr
                    anomaly_mask=rotate_image(anomaly_mask)
                    img_thr = anomaly_img_augmented.astype(np.float32) * anomaly_mask
                    image_=image.astype(np.float32) * (1-anomaly_mask)
                    augmented_image = image_ +img_thr
                    x.append(augmented_image)
                    mask.append(image_path)
                    n+=1
                    
                if img_type == 'good':
                    y.extend([0] * n)#标签为0
#                     mask.extend([-1] * n)
                else:
                    y.extend([1] * n)#标签为1
#                     mask.extend([-1] * n)
                x,y,mask=x, list(y), list(mask)
        assert len(x) == len(y), 'number of x and y should be same'
        return x, y, mask

    def __len__(self):
        return len(self.x)

#     def randAugmenter(self):
#         aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
#         aug = iaa.Sequential([self.augmenters[aug_ind[0]],
#                               self.augmenters[aug_ind[1]],
#                               self.augmenters[aug_ind[2]]]
#                              )
#         return aug
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
            x.extend(img_fpath_list)

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
