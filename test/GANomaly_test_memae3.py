# -*- coding:utf8 -*-
# @TIME     : 2020/12/10 16:01
# @Author   : SuHao
# @File     : GANomaly_test.py

import os
import tqdm
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T

import sys
sys.path.append("..")
from dataload.dataload import load_dataset
from models.DCGAN_GANomaly_ori import NetG, NetD
from evaluate import Evaluate, draw_heatmap
# from models.GANomaly_transformer import NetG
# from models.Ganomaly_SE2 import NetG,NetD
from models.memory_module import EntropyLossEncap 
# from models.SSIM import SSIM
# from models.ml_memAE_sc import NetG,NetD
from models.Unet4_w2_add_pill import NetG,NetD
# from models.DCGAN_GANomaly_ori import NetG, NetD
#MainboardScrew05
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/test_Unet4_w2_add_pill__ssim_bigshrink_D", help="path to save experiments results")
parser.add_argument("--dataset", default="pill", help="mnist")
parser.add_argument('--dataroot', default=r"../data/", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=1, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--size", type=int, default=128, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=128, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--n_extra_layers", type=int, default=0, help="extra layers of Encoder and Decoder")
parser.add_argument("--gen_pth", default=r"../experiments/train_Unet4_w2_add_pill__ssim_bigshrink_D/gen_best.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/train_Unet4_w2_add_pill__ssim_bigshrink_D/disc_best.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.size, trans=None, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
opt.dataSize = test_dataset.__len__()

## model
gen = NetG(opt).to(device)
disc = NetD(opt).to(device)
assert gen.load_state_dict(torch.load(opt.gen_pth))
assert disc.load_state_dict(torch.load(opt.disc_pth))
print("Pretrained models have been loaded.")

## record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

## def
def splitImage(img, size):#128,128
    if img.size(3) % size !=0:
        return
    num = int(img.size(3) / size)
    results = torch.zeros(num**2, img.size(1), size, size)#1,3,128,128
    split1 = torch.split(img, size, dim=2)
    for i in range(num):
        split2 = torch.split(split1[i], size, dim=3)
        for j in range(num):
            results[i*num+j, :, :, :] = split2[j]
    return results


def catImage(imgs, size):#[1, 3, 128, 128],(1,1)
    if imgs.size(0) != size[0] * size[1]:
        return
    results = torch.zeros(1, imgs.size(1), imgs.size(2)*size[0], imgs.size(3)*size[1])
    width = imgs.size(2)
    height = imgs.size(3)
    for i in range(size[0]):
        for j in range(size[1]):
            results[0, :, i*width:(i+1)*width, j*height:(j+1)*height] = imgs[i*size[0]+j]
#             print(imgs[i*size[0]+j].shape)#torch.Size([3, 128, 128])
    return results


def load(img):
    transform_x = T.Compose([T.ToTensor()])
#     transform_mask = T.Compose([T.ToTensor(),])
    x=Image.fromarray(img).convert('RGB')
    x = transform_x(x)
    return x


## loss
# L_con = nn.L1Loss(reduction='mean')
L_lat = nn.MSELoss(reduction='mean')
L_con = nn.MSELoss(reduction='mean')

L_con_ = nn.MSELoss(reduction='mean')

# loss_ssim = SSIM()
## test
gen.eval()
disc.eval()
con_loss = []
lat_loss = []
# ssim_loss=[]
# total_loss = []
labels = []
# score=[]
total_loss=[[]for z in range(66)]
lambda_=[]
res_loss=[]

# grid  lambda_L_con=0.1, lambda_L_lat=0.9
# bottle  lambda_L_con=1, lambda_L_lat=0

# lambda_L_con=0.4
# lambda_L_lat=0.6

evaluation = Evaluate(opt.experiment)
tqdm_loader = tqdm.tqdm(test_dataloader)
for i, (test_input, label, mask) in enumerate(tqdm_loader):
    tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
    # 1 3 128 128
#     print('test_input.size(3)',test_input.size(0),test_input.size(1),test_input.size(2),test_input.size(3))
    patches_num = int(test_input.size(3) / opt.imageSize)
#     print('splitImage(test_input, opt.imageSize)',(splitImage(test_input, opt.imageSize)).shape)# torch.Size([1, 3, 128, 128])
    test_inputs = splitImage(test_input, opt.imageSize).to(device)#128,128
    ## inference
    with torch.no_grad():
        outputs,_,_,_,_,_,_,_,_= gen(test_inputs)
        _, features_real = disc(test_inputs)
        _, features_feak = disc(outputs)
    con = L_con(outputs, test_inputs).item()
#     ssim_loss = loss_ssim(outputs, test_inputs).item()
    lat = L_lat(features_real, features_feak).item()
    
    
    labels.append(label.item())
    
    output = catImage(outputs, (patches_num, patches_num))
#     print('output',output.shape)
    residule = torch.abs(test_input - output)
 #     print('+++++++++++++++++')
#     print('residule1',residule.shape)#torch.Size([1, 3, 128, 128])
    vutils.save_image(torch.cat((test_input, output, residule), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i))
#     vutils.save_image(output, '{0}/{1}-output.png'.format(opt.experiment, i))
#     vutils.save_image(residule, '{0}/{1}-residule.png'.format(opt.experiment, i))
#     vutils.save_image(test_input, '{0}/{1}-test_input.png'.format(opt.experiment, i))
    residule_=residule
    residule_ = residule_.detach().cpu().numpy()
    residule_ = draw_heatmap(residule_)
    cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule_)
    
    
#     x=torch.tensor([10.0])
    residule = (residule - torch.max(residule)) / (torch.max(residule) - torch.min(residule))
    
#     print(x)
    residule=residule.to(device)
    y= torch.zeros([1,3, 128,128])
    y = y.to(device)#128,128
    res=L_con_(residule,y).item()
    res_loss.append(res)
#     t=0
#     for j in range(0,11):
#         for z in range(11-j):
#             for k in range(11-(j+z)):
#                 lambda_L_con=j
#                 lambda_L_lat=z
#                 m=k
#                 score=con*lambda_L_con + lat*lambda_L_lat+m*res
#                 total_loss[t].append(score)
#                 t+=1
    
#     for j in range(0,11):
#         for z in range(11):
#             for k in range(11):
#                 lambda_L_con=j
#                 lambda_L_lat=z
#                 m=k
#                 score=con*lambda_L_con + lat*lambda_L_lat+m*res
#                 total_loss[t].append(score)
#                 t+=1
# print(len(total_loss))
# print(len(total_loss[1]))




    t=0
    for j in range(0,11):
        for z in range(11-j):
            k =10-(j+z)
            score=con*j + lat*z+k*res
            total_loss[t].append(score)
            t+=1
            
for j in range(0,11):
        for z in range(11-j):
            k =10-(j+z)
            n=(j,z,k)
            lambda_.append(n)
            
print(len(total_loss))
print(len(total_loss[1]))

#     residule = draw_heatmap(residule)
#     cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule)
    
    
# print(enc_loss)
# print(labels)
# enc_loss = np.array(enc_loss)
# enc_loss = (enc_loss - np.min(enc_loss)) / (np.max(enc_loss) - np.min(enc_loss))
# print('enc_loss',enc_loss)
# evaluation.labels = labels
# evaluation.scores = enc_loss
# ##############
# evaluation.run()    
    
    
 # print(score)
# print(labels)
# score = np.array(res_loss)
# # score = (score - np.min(score)) / (np.max(score) - np.min(score))
# print('score',score)
# evaluation.labels = labels
# evaluation.scores = score
# ##############
# auc=evaluation.run()
  
    
# print(score)
# print(labels)
# score = np.array(total_loss)
# score = (score - np.min(score)) / (np.max(score) - np.min(score))
# print('score',score)
# evaluation.labels = labels
# evaluation.scores = score
# #############
# evaluation.run()

# 
print("==============start==================")
m=0
n=[]
lambda_L_con=0
lambda_L_lat=0
lambda_hot=0
for i in range(len(total_loss)):
#     print(total_loss[i])
    score = np.array(total_loss[i])
    score = (score - np.min(score)) / (np.max(score) - np.min(score))
    #     print('score',score)
    evaluation.labels = labels
    evaluation.scores = score
    ##############
    auc=evaluation.run()
    if m<auc[0]:
        m=auc[0]
        n= auc
        lambda_L_con,lambda_L_lat,lambda_hot=lambda_[i]
#         print(lambda_[i])

print("auc: ", n[0])
print("best_accuracy: ", n[1])
print("best_thre: ",  n[2])
print("best_F1_score: ",  n[3])
print("lambda_L_con:{},lambda_L_lat:{},lambda_hot:{} ".format(lambda_L_con,lambda_L_lat,lambda_hot))
print("==============end==================")






