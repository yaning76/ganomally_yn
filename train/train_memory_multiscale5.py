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
from models.memory_multiscale6 import NetG,NetD
from models.SSIM import SSIM

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/train_memory_multiscale6_cable", help="path to save experiments results")
parser.add_argument("--dataset", default="cable", help="mnist")
parser.add_argument('--dataroot', default=r"../data/", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=800, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=256, help="size of each image dimension")
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
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

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
L_con0 = nn.L1Loss()         # For gen

L_lat = nn.MSELoss()        # For gen
L_adv_disc = nn.BCELoss()        # For disc
tr_entropy_loss_func0 = EntropyLossEncap().to(device)
tr_entropy_loss_func1 = EntropyLossEncap().to(device)
tr_entropy_loss_func2= EntropyLossEncap().to(device)
tr_entropy_loss_func3 = EntropyLossEncap().to(device)
# tr_entropy_loss_func4 = EntropyLossEncap().to(device)
# tr_entropy_loss_func5 = EntropyLossEncap().to(device)
# tr_entropy_loss_func6 = EntropyLossEncap().to(device)
# tr_entropy_loss_func7 = EntropyLossEncap().to(device)
# tr_entropy_loss_func8 = EntropyLossEncap().to(device)
# tr_entropy_loss_func9 = EntropyLossEncap().to(device)
# tr_entropy_loss_func10 = EntropyLossEncap().to(device)

loss_ssim = SSIM()
tolloss=10000
entropy_loss_weight=20
## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        L_adv_gen_epoch_loss = 0.0
        L_con_epoch_loss = 0.0
        L_con_epoch_loss0 = 0.0
        
        L_lat_epoch_loss = 0.0
        L_ssim=0.0
        L_tr_entropy_loss_func0 = 0.0
        L_tr_entropy_loss_func1 = 0.0
        L_tr_entropy_loss_func2 = 0.0
        L_tr_entropy_loss_func3 = 0.0
#         L_tr_entropy_loss_func4 = 0.0
#         L_tr_entropy_loss_func5 = 0.0
#         L_tr_entropy_loss_func6 = 0.0
#         L_tr_entropy_loss_func7 = 0.0
#         L_tr_entropy_loss_func8 = 0.0
#         L_tr_entropy_loss_func9 = 0.0
#         L_tr_entropy_loss_func10 = 0.0
        
#         L_enc_epoch_loss = 0.0
        L_total_epoch_loss = 0.0
        disc_epoch_loss = 0.0
#         l_en=[]
        for inputs, _, _ in train_dataloader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
#             print(inputs.shape)
            #label_real tensor([1., 1., 1., 1., 1., ...],device='cuda:0')
            label_real = torch.ones(batch_size).to(device)
#             print('label_real',label_real)
            label_fake = torch.zeros(batch_size).to(device)
#             print('label_fake',label_fake)
########################################################
            ## Update "D": max log(D(x)) + log(1-D(G(z))
            disc_optimizer.zero_grad()
        
            D_real, _ = disc(inputs)
            #torch.Size([32]) torch.Size([32])
#             print(D_real.shape,label_real.shape)
            disc_loss_real = L_adv_disc(D_real, label_real.detach())#//
            
            outputs0,_,outputs= gen(inputs)
#             print(outputs.shape)
            D_fake, _ = disc(outputs)#//
            disc_loss_fake =L_adv_disc(D_fake, label_fake)
            
            disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
            
            disc_loss.backward()
            disc_optimizer.step()
            disc_epoch_loss += disc_loss.item() * batch_size
########################################################

            ## Update 'G' : max log(D(G(z)))
            gen_optimizer.zero_grad()
        #torch.Size([32, 3, 128, 128]) torch.Size([32, 100, 1, 1]) torch.Size([32, 100, 1, 1])
            outputs0, res,outputs= gen(inputs)
#             print(outputs.shape, latent_in.shape, latent_out.shape)#
            D_fake, feature_fake = disc(outputs)
#             print(feature_fake.shape)
            _, feature_real = disc(inputs)
            adv_loss = L_adv_gen(D_fake, label_real.detach())
            con_loss = L_con(outputs, inputs)
            con_loss0 = L_con0(outputs0, inputs)#++++++
            
            ssim_loss = loss_ssim(outputs, inputs)###
            enc_loss = L_lat(feature_fake, feature_real)
            entropy_loss0=tr_entropy_loss_func0(res[0]["att"])
            entropy_loss1 = tr_entropy_loss_func1(res[1]["att"])
            entropy_loss2 = tr_entropy_loss_func2(res[2]["att"])
            entropy_loss3 = tr_entropy_loss_func3(res[3]["att"])
#             entropy_loss4 = tr_entropy_loss_func4(res[4]["att"])
#             entropy_loss5 = tr_entropy_loss_func5(res[5]["att"])
#             entropy_loss6 = tr_entropy_loss_func6(res[6]["att"])
#             entropy_loss7 = tr_entropy_loss_func7(res[7]["att"])
#             entropy_loss8 = tr_entropy_loss_func8(res[8]["att"])
#             entropy_loss9 = tr_entropy_loss_func9(res[9]["att"])
#             entropy_loss10 = tr_entropy_loss_func10(res[10]["att"])
            entropy_loss=entropy_loss0+entropy_loss1+entropy_loss2+entropy_loss3

#             enc_loss = L_enc(latent_out, latent_in)
            b,c,h,w=inputs.shape
#             print(enc_loss/b)
#             l_en.append(enc_loss)
            
            total_loss = opt.lambda_adv*adv_loss + \
        opt.lambda_con *(con_loss+ssim_loss+con_loss0) + \
        opt.lambda_lat*enc_loss+ \
        entropy_loss_weight*entropy_loss
            total_loss.backward()
            L_adv_gen_epoch_loss += adv_loss.item() * batch_size
            L_con_epoch_loss += con_loss.item() * batch_size
            L_con_epoch_loss0 += con_loss0.item() * batch_size
            
            L_lat_epoch_loss += enc_loss.item() * batch_size
            L_ssim+=ssim_loss.item() * batch_size
            L_tr_entropy_loss_func0+=entropy_loss0.item() * batch_size
            L_tr_entropy_loss_func2+=entropy_loss2.item() * batch_size
            L_tr_entropy_loss_func1+=entropy_loss1.item() * batch_size
            L_tr_entropy_loss_func3+=entropy_loss3.item() * batch_size
#             L_tr_entropy_loss_func4+=entropy_loss4.item() * batch_size
#             L_tr_entropy_loss_func5+=entropy_loss5.item() * batch_size
#             L_tr_entropy_loss_func6+=entropy_loss6.item() * batch_size
#             L_tr_entropy_loss_func7+=entropy_loss7.item() * batch_size
#             L_tr_entropy_loss_func8+=entropy_loss8.item() * batch_size
#             L_tr_entropy_loss_func9+=entropy_loss9.item() * batch_size
#             L_tr_entropy_loss_func10+=entropy_loss10.item() * batch_size
            
#             L_enc_epoch_loss += enc_loss.item() * batch_size
            L_total_epoch_loss += total_loss.item() * batch_size
            
#             disc_optimizer.step()
            gen_optimizer.step()

            ## record results
#             if record % opt.sample_interval == 0:
#             # outputs.data = outputs.data.mul(0.5).add(0.5)
#                 vutils.save_image(outputs.view(-1, opt.nc, opt.imageSize, opt.imageSize),
#                                   '{0}/outputs_{1}.png'.format(opt.experiment, record))
# #                 vutils.save_image(inputs.view(-1, opt.nc, opt.imageSize, opt.imageSize),
# #                                   '{0}/inputs_{1}.png'.format(opt.experiment, record))
#             record += 1

#         print(enc_loss)
        ## End of epoch
        L_adv_gen_epoch_loss /= opt.dataSize
        L_con_epoch_loss /= opt.dataSize
        L_con_epoch_loss0 /= opt.dataSize
        
        L_lat_epoch_loss /= opt.dataSize
        L_ssim/= opt.dataSize
        L_tr_entropy_loss_func0/= opt.dataSize
        L_tr_entropy_loss_func2 /= opt.dataSize
        L_tr_entropy_loss_func1 /= opt.dataSize
        L_tr_entropy_loss_func3 /= opt.dataSize
#         L_tr_entropy_loss_func4 /= opt.dataSize
#         L_tr_entropy_loss_func5 /= opt.dataSize
#         L_tr_entropy_loss_func6 /= opt.dataSize
#         L_tr_entropy_loss_func7 /= opt.dataSize
#         L_tr_entropy_loss_func8 /= opt.dataSize
#         L_tr_entropy_loss_func9 /= opt.dataSize
#         L_tr_entropy_loss_func10 /= opt.dataSize
        
        
        L_total_epoch_loss /= opt.dataSize
        disc_epoch_loss /= opt.dataSize
        
        
        print('L_adv:{},L_con:{},L_lat_epoch_loss:{},L_tr_entropy_loss_func:{},L_total:{},disc_loss{}'.format(L_adv_gen_epoch_loss,L_con_epoch_loss,L_lat_epoch_loss,L_tr_entropy_loss_func1,L_total_epoch_loss,disc_epoch_loss))
#         t.set_postfix(L_adv=L_adv_epoch_loss, L_con=L_con_epoch_loss, L_enc=L_enc_epoch_loss,
#                       L_total=L_total_epoch_loss, disc_loss=disc_epoch_loss)
        writer.add_scalar("gen_epoch_loss", L_total_epoch_loss, e)
        writer.add_scalar("disc_epoch_loss", disc_epoch_loss, e)

#     if (e+1) % 10 == 0:
        if tolloss>L_total_epoch_loss:
            tolloss=L_total_epoch_loss
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_best.pth'.format(opt.experiment))
            torch.save(disc.state_dict(), '{0}/disc_best.pth'.format(opt.experiment))
        torch.save(gen.state_dict(), '{0}/gen.pth'.format(opt.experiment))
        torch.save(disc.state_dict(), '{0}/disc.pth'.format(opt.experiment))
writer.close()