# -*- coding:utf8 -*-
# @TIME     : 2020/12/10 10:28
# @Author   : SuHao
# @File     : GANomaly_trian.py

from __future__ import print_function
import os
import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append("..")
from dataload.dataload import load_dataset
# from models.Ganomaly_SSPCAB2_1 import NetG,NetD
# from models.NetD import NetD
from models.memory_module import EntropyLossEncap
from models.mem import NetG,NetD
from models.SSIM import SSIM

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/train_mem_toothbrush", help="path to save experiments results")
parser.add_argument("--dataset", default="toothbrush", help="mnist")
parser.add_argument('--dataroot', default=r"../data/", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=400, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=128, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--n_extra_layers", type=int, default=0, help="extra layers of Encoder and Decoder")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--lambda_adv", type=int, default=1, help="weight of loss_adv")
parser.add_argument("--lambda_con", type=int, default=40, help="weight of loss_con")
parser.add_argument("--lambda_lat", type=int, default=1, help="weight of loss_enc")
parser.add_argument("--gen_pth", default=r"../experiments/a/gen_best.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/a/disc_best.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# random seed
opt.seed = 42
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
train_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=None, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)

## model init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


## model
gen = NetG(opt).to(device)
disc = NetD(opt).to(device)
# gen = Generator(opt.imageSize, opt.nz, opt.nc).to(device)
# disc = Discriminator(opt.imageSize, opt.nc).to(device)
# gen.apply(weights_init)
# disc.apply(weights_init)
try:
    gen.load_state_dict(torch.load(opt.gen_pth))
    disc.load_state_dict(torch.load(opt.disc_pth))
    print("Pretrained models have been loaded.")
except:
    print("Pretrained models fail.")

## record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1).to(device)

## opt.dataSize
opt.dataSize = train_dataset.__len__()

## loss function
gen_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
disc_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
L_adv_gen = nn.MSELoss()        # For gen
L_con = nn.L1Loss()         # For gen
L_lat = nn.MSELoss()        # For gen
L_adv_disc = nn.BCELoss()        # For disc
tr_entropy_loss_func2 = EntropyLossEncap().to(device)
tr_entropy_loss_func3 = EntropyLossEncap().to(device)
tr_entropy_loss_func2_ = EntropyLossEncap().to(device)
tr_entropy_loss_func3_ = EntropyLossEncap().to(device)
tr_entropy_loss_func1 = EntropyLossEncap().to(device)
tr_entropy_loss_func1_ = EntropyLossEncap().to(device)
tr_entropy_loss_func0 = EntropyLossEncap().to(device)
tr_entropy_loss_func0_ = EntropyLossEncap().to(device)
loss_ssim = SSIM()
tolloss=10000
entropy_loss_weight=0.02#0.002
## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        L_con_epoch_loss = 0.0
        L_tr_entropy_loss_func3 = 0.0
        
        for inputs, _, _ in train_dataloader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            outputs, att3= gen(inputs)
            con_loss = L_con(outputs, inputs)
            recon_loss_val = con_loss.item()
            entropy_loss = tr_entropy_loss_func3(att3)
            entropy_loss_val = entropy_loss.item()
            loss = con_loss + 0.002 * entropy_loss
            loss_val = loss.item()
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()
            
               
        
        print('loss:{},L_con:{},entropy_loss:{}'.format(loss,con_loss,entropy_loss))
#         t.set_postfix(L_adv=L_adv_epoch_loss, L_con=L_con_epoch_loss, L_enc=L_enc_epoch_loss,
#                       L_total=L_total_epoch_loss, disc_loss=disc_epoch_loss)
        writer.add_scalar("gen_epoch_loss", loss, e)

#     if (e+1) % 10 == 0:
        if tolloss>loss:
            tolloss=loss
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_best.pth'.format(opt.experiment))
        torch.save(gen.state_dict(), '{0}/gen.pth'.format(opt.experiment))
writer.close()