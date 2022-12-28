# -*- coding:utf8 -*-
# @TIME     : 2020/12/10 16:01
# @Author   : SuHao
# @File     : GANomaly_test.py

import os
import tqdm
import torch
import numpy as np
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import sys
sys.path.append("..")
from dataload.dataload_yn import load_dataset
from models.DCGAN_GANomaly_ori import NetG, NetD
from evaluate import Evaluate, draw_heatmap
# from models.GANomaly_transformer import NetG
# from models.Ganomaly_SE2 import NetG,NetD
from models.memory_module import EntropyLossEncap 
from models.SSIM import SSIM
# from models.ml_memAE_sc import NetG,NetD
from models.SCADN1 import NetG,NetD
# from models.DCGAN_GANomaly_ori import NetG, NetD
#MainboardScrew05
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/test_SCADN1_32", help="path to save experiments results")
parser.add_argument("--dataset", default="capsule", help="mnist")
parser.add_argument('--dataroot', default=r"../data/", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=1, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--size", type=int, default=256, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=256, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--n_extra_layers", type=int, default=0, help="extra layers of Encoder and Decoder")
parser.add_argument("--gen_pth", default=r"../experiments/train_SCADN1_32/gen_best.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/train_SCADN1_32/disc_best.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

## loss
# L_con = nn.L1Loss(reduction='mean')
L_lat = nn.MSELoss(reduction='mean')
L_con = nn.MSELoss(reduction='mean')
loss_ssim = SSIM()
## test
gen.eval()
disc.eval()
con_loss = []
lat_loss = []
# ssim_loss=[]
# total_loss = []
labels = []
# score=[]
total_loss=[[]for z in range(11)]
# grid  lambda_L_con=0.1, lambda_L_lat=0.9
# bottle  lambda_L_con=1, lambda_L_lat=0

# lambda_L_con=0.4
# lambda_L_lat=0.6

evaluation = Evaluate(opt.experiment)
tqdm_loader = tqdm.tqdm(test_dataloader)
for i, (test_input, label, mask) in enumerate(tqdm_loader):
    tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
    mask = mask.to(device)
    test_img1=test_input[0].to(device)
    test_img2=test_input[1].to(device)
    with torch.no_grad():
        output,_= gen(test_img1,test_img2)
        _, features_real = disc(mask)
        _, features_feak = disc(output)
#             torch.cat((mask, output, residule), dim=0)
        vutils.save_image(output, '{0}/{1}.png'.format(opt.experiment, i))
    vutils.save_image(mask, '{0}/{1}-mask.png'.format(opt.experiment, i))
#     # 1 3 128 128
# #     print('test_input.size(3)',test_input.size(0),test_input.size(1),test_input.size(2),test_input.size(3))
#     patches_num = int(test_input.size(3) / opt.imageSize)
# #     print('splitImage(test_input, opt.imageSize)',(splitImage(test_input, opt.imageSize)).shape)# torch.Size([1, 3, 128, 128])
#     test_inputs = splitImage(test_input, opt.imageSize).to(device)#128,128
#     mask = mask.to(device)
# #     print(test_inputs)
#     ## inference
#     with torch.no_grad():
#         outputs= gen(test_inputs)
#         _, features_real = disc(mask)
#         _, features_feak = disc(outputs)
#     con = L_con(outputs, mask).item()
#     ssim_loss = loss_ssim(outputs, mask).item()
#     con=con+ssim_loss
#     lat = L_lat(features_real, features_feak).item()
    
#     for j in range(0,11):
#         lambda_L_con=j*0.1
#         lambda_L_lat=(10-j)*0.1
#         score=con*lambda_L_con + lat*lambda_L_lat
        
#         total_loss[j].append(score)
# #         print(total_loss[j])
#         #torch.Size([1, 3, 128, 128]) torch.Size([1, 100, 1, 1]) torch.Size([1, 100, 1, 1])
# #         print('outputs.shape, latent_in.shape, latent_out.shape',outputs.shape, latent_in.shape, latent_out.shape)
# #     enc_loss.append(L_enc(latent_in, latent_out).item())
#     ############################################################
# #     score.append(total_loss)
    
#     labels.append(label.item())
    
#     output = catImage(outputs, (patches_num, patches_num))
#     mask = catImage(mask, (patches_num, patches_num))
    
# #     print('output',output.shape)
#     residule = torch.abs(mask - output)
# #     print('residule',residule.shape)#torch.Size([1, 3, 128, 128])
#     vutils.save_image(torch.cat((mask, output, residule), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i))
# #     vutils.save_image(output, '{0}/{1}-output.png'.format(opt.experiment, i))
# #     vutils.save_image(residule, '{0}/{1}-residule.png'.format(opt.experiment, i))
# #     vutils.save_image(test_input, '{0}/{1}-test_input.png'.format(opt.experiment, i))

#     residule = residule.detach().cpu().numpy()
#     residule = draw_heatmap(residule)
#     cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule)
    
    
# # print(enc_loss)
# # print(labels)
# # enc_loss = np.array(enc_loss)
# # enc_loss = (enc_loss - np.min(enc_loss)) / (np.max(enc_loss) - np.min(enc_loss))
# # print('enc_loss',enc_loss)
# # evaluation.labels = labels
# # evaluation.scores = enc_loss
# # ##############
# # evaluation.run()    
    
    
    
    
# # # print(score)
# # print(labels)
# # score = np.array(total_loss)
# # score = (score - np.min(score)) / (np.max(score) - np.min(score))
# # print('score',score)
# # evaluation.labels = labels
# # evaluation.scores = score
# # ##############
# # evaluation.run()

# print("==============start==================")
# m=0
# n=[]
# lambda_L_con=0
# lambda_L_lat=0
# for i in range(11):
# #     print(total_loss[i])
#     score = np.array(total_loss[i])
#     score = (score - np.min(score)) / (np.max(score) - np.min(score))
#     #     print('score',score)
#     evaluation.labels = labels
#     evaluation.scores = score
#     ##############
#     auc=evaluation.run()
#     if m<auc[0]:
#         m=auc[0]
#         n= auc
#         lambda_L_con=i*0.1
#         lambda_L_lat=(10-i)*0.1

# print("auc: ", auc[0])
# print("best_accuracy: ", auc[1])
# print("best_thre: ",  auc[2])
# print("best_F1_score: ",  auc[3])
# print("lambda_L_con:{},lambda_L_lat:{} ".format(lambda_L_con,lambda_L_lat))
# print("==============end==================")






