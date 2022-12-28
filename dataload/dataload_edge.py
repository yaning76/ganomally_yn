# -*- coding:utf8 -*-
# @TIME     : 2020/12/7 10:12
# @Author   : SuHao
# @File     : dataload.py

import torchvision.transforms as T
import torchvision.datasets as dset
from dataload.mvtec_edge import MVTecDataset_edge

def load_dataset(dataroot, dataset_name, imageSize, trans, train=True):
    params_med = {"dataroot": dataroot, "split": 'train' if train else 'test', "transform":trans}
    if dataset_name == "mnist":
        dataset = dset.MNIST(root=dataroot,
                             train=train,
                             download=True,
                             transform=T.Compose([T.Resize(imageSize), T.ToTensor()]),
                             )
    else:
        dataset = MVTecDataset_edge(dataroot, class_name=dataset_name, is_train=train, resize=imageSize)
    return dataset