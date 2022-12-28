import math
# from models.basic_modules import *
import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule,MemModule1

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2),
            # double_conv(in_ch, out_ch)

            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            double_conv(out_ch, out_ch),

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, 1), )
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)

        self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x = self.conv(x1)
        return x
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
        base_channels=64
        mem_dim=2000
        shrink_thres=0.0025
        
        self.initial0 = double_conv(nc, base_channels)#128
        self.pyramid0 = down(base_channels, base_channels*2)#64
        self.pyramid1 = down(base_channels*2, base_channels*4)#32
        self.pyramid2 = down(base_channels*4, base_channels*8)#16
        self.pyramid3 = down(base_channels*8, base_channels*16)#8
        
        self.mem_rep3 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*16, shrink_thres =shrink_thres)
        self.mem_rep3_ = MemModule1(mem_dim=mem_dim, fea_dim=64, shrink_thres =shrink_thres)
#         self.conv3 = double_conv(base_channels*32,base_channels*16)
        
        self.mem_rep2 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*8, shrink_thres =shrink_thres)
        self.mem_rep2_ = MemModule1(mem_dim=mem_dim, fea_dim=256, shrink_thres =shrink_thres)
#         self.conv2 = double_conv(base_channels*16,base_channels*8)
        self.conv2_ = double_conv(base_channels*16,base_channels*8)
    
        self.mem_rep1 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*4, shrink_thres =shrink_thres)
        self.mem_rep1_ = MemModule1(mem_dim=mem_dim, fea_dim=1024, shrink_thres =shrink_thres)
#         self.conv2 = double_conv(base_channels*16,base_channels*8)
        self.conv1_ = double_conv(base_channels*8,base_channels*4)
        
        self.pyramid0_D = up(base_channels*16, base_channels*8)#32
        self.pyramid1_D = up(base_channels*8, base_channels*4)#32
        self.pyramid2_D = up(base_channels*4, base_channels*2)#64
        self.pyramid3_D = up(base_channels*2, base_channels)#128
        
        self.pyramid4_D =nn.Conv2d(base_channels, nc, 1)
        
    def forward(self, input):
#Encoder
        out=self.initial0(input)
        out0=self.pyramid0(out)
        out1=self.pyramid1(out0)
        out2=self.pyramid2(out1)
        out3=self.pyramid3(out2)
#         print(out3.shape)
        res_mem3= self.mem_rep3(out3)
        f3 = res_mem3['output']
        att3 = res_mem3['att']
        
        res_mem3_= self.mem_rep3_(out3)
        f3_ = res_mem3_['output']
        att3_ = res_mem3_['att']
#         print()
#         x = torch.cat([f3, f3_], dim=1)
        x=f3+f3_
#         x=self.conv3(x)
        
        res_mem2= self.mem_rep2(out2)
        f2 = res_mem2['output']
        att2 = res_mem2['att']
        res_mem2_= self.mem_rep2_(out2)
        f2_ = res_mem2_['output']
        att2_ = res_mem2_['att']
        f2=f2+f2_
        res_mem1= self.mem_rep1(out1)
        f1 = res_mem1['output']
        att1 = res_mem1['att']
        res_mem1_= self.mem_rep1_(out1)
        f1_ = res_mem1_['output']
        att1_ = res_mem1_['att']
        f1=f1+f1_
#         res_mem1 = self.mem_rep1(out1)
#         f1 = res_mem1['output']
#         att1 = res_mem1['att']
#Decoder
        x=self.pyramid0_D(x)
#         x = torch.cat([x, f2], dim=1)
#         x=self.conv2(x)
        x = torch.cat([f2,x], dim=1)
        x=self.conv2_(x)
        x=self.pyramid1_D(x)
        x = torch.cat([f1,x], dim=1)
        x=self.conv1_(x)
#         res_mem0 = self.mem_rep0(x)
#         f0 = res_mem0['output']
#         att0 = res_mem0['att']
        x=self.pyramid2_D(x)
        x=self.pyramid3_D(x)
        x=self.pyramid4_D(x)
        
#         print(x.shape)
        return x,att3,att3_,att2,att2_,att1,att1_
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    def forward(self, x):
        gen_imag,att3,att3_,att2,att2_,att1,att1_= self.encoder1(x)
        return gen_imag,att3,att3_,att2,att2_,att1,att1_
    
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ngf = 64
        self.model = nn.Sequential(
            nn.Conv2d(opt.nc, ngf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf << 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 1),       # 128

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 1, ngf << 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 2),       # 256

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 2, ngf << 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 3),       # 512
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 3, ngf << 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 4),       # 1024
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 4, 100, 4, 1, 0, bias=False),       # 512
            nn.BatchNorm2d(100),       # 100
        )

        self.classify = nn.Sequential(
            nn.Conv2d(100, 1, 3, 1, 1, bias=False),       # 512
            nn.Sigmoid(),
        )


    def forward(self, x):
        feature = self.model(x)
        classification = self.classify(feature)
        return classification.view(-1, 1).squeeze(1), feature    
    
