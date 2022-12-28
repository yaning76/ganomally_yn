import math
# from models.basic_modules import *
import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule_window,MemModule,MemModule1_new
# MemModule1_new  h*w h/window*w/window
# MemModule_window  widow*window*c  (widow*window*c)/c_size       window=2,c_size=2
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
#         print(x.shape)
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2),
#             double_conv(in_ch, out_ch)

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

        self.conv = double_conv(in_ch // 2, out_ch)

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
        shrink_thres=0.00025
        window_size=2
        
        self.initial0 = double_conv(nc, base_channels)#64,256,256
        self.down0 = nn.MaxPool2d(kernel_size=2)#128,128,128
        self.en_doubleConv1 = double_conv(base_channels, base_channels*2)
        self.down1 = nn.MaxPool2d(kernel_size=2)#128,128,128
        self.en_doubleConv2 = double_conv(base_channels*2, base_channels*4)
        self.down2 = nn.MaxPool2d(kernel_size=2)#128,128,128
        self.en_doubleConv3 = double_conv(base_channels*4, base_channels*8)
        self.down3 = nn.MaxPool2d(kernel_size=2)#128,128,128
        self.en_doubleConv4 = double_conv(base_channels*16, base_channels*16)
        
        self.up1 = nn.ConvTranspose2d(in_channels=base_channels*16, out_channels=base_channels*8, kernel_size=2, stride=2)
        self.de_doubleConv1 = double_conv(base_channels*20, base_channels*8)
        self.up2 = nn.ConvTranspose2d(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=2, stride=2)
        self.de_doubleConv2 = double_conv(base_channels*10, base_channels*4)
        self.up3 = nn.ConvTranspose2d(in_channels=base_channels*4, out_channels=base_channels*2, kernel_size=2, stride=2)
        self.de_doubleConv3 = double_conv(base_channels*5, base_channels*2)
        
        self.up4 = nn.ConvTranspose2d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=2, stride=2)
        self.de_doubleConv4 = double_conv(base_channels, base_channels)

# MemModule_window  (mem_dim, fea_dim,window,c_num,)
# window 窗口大小, c_num 通道比例 fea_dim=c*window *window /c_num
# MemModule1_new(mem_dim,fea_dim,window=1,)
# window 窗口比例 fea_dim=w/window*h/window

        ############               
        self.mem_rep2_0 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*4, window=2,c_num=8,shrink_thres =shrink_thres)
        self.mem_rep2_1 = MemModule1_new(mem_dim=mem_dim, fea_dim=256, window=1,shrink_thres =shrink_thres)
        ############               
        self.mem_rep1_0 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*2*4, window=4,c_num=8,shrink_thres =shrink_thres)
        self.mem_rep1_1 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*4*4, window=8,c_num=16,shrink_thres =shrink_thres)
        self.mem_rep1_2 = MemModule1_new(mem_dim=mem_dim, fea_dim=1024,window=1, shrink_thres =shrink_thres)
        ############                       
        self.mem_rep0_0 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*2*8, window=8,c_num=8,shrink_thres =shrink_thres)
        self.mem_rep0_1 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*2*8, window=16,c_num=32,shrink_thres =shrink_thres)
    
        self.mem_rep0_2 = MemModule1_new(mem_dim=mem_dim, fea_dim=1024,window=2, shrink_thres =shrink_thres)
    ############ 
        self.mem_rep_0 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*16, window=16,c_num=16,shrink_thres =shrink_thres)
        self.mem_rep_1 = MemModule_window(mem_dim=mem_dim, fea_dim=base_channels*32, window=32,c_num=32,shrink_thres =shrink_thres)
        
        self.mem_rep_2 = MemModule1_new(mem_dim=mem_dim, fea_dim=1024,window=4, shrink_thres =shrink_thres)
        
       
        self.pyramid4_D =nn.Conv2d(base_channels, nc, 1)
        
    def forward(self, input):
    #Encoder
#         f=[]
        res=[]
        out=self.initial0(input)
        out=self.down0(out)#64,128,128
        
        res_0=self.mem_rep_0(out)
        res_1=self.mem_rep_1(out)
        res_2=self.mem_rep_2(out)
        res.append(res_0)
        res.append(res_1)
        res.append(res_2)
        
        out=self.en_doubleConv1(out)
        out=self.down1(out)
        res0_0=self.mem_rep0_0(out)
        res0_1=self.mem_rep0_1(out)
        res0_2=self.mem_rep0_2(out)
        res.append(res0_0)
        res.append(res0_1)
        res.append(res0_2)
        
        out=self.en_doubleConv2(out)
        out=self.down2(out)
        res1_0=self.mem_rep1_0(out)
        res1_1=self.mem_rep1_1(out)
        res1_2=self.mem_rep1_2(out)
        res.append(res1_0)
        res.append(res1_1)
        res.append(res1_2)
        
        out=self.en_doubleConv3(out)
        out=self.down3(out)
        res2_0=self.mem_rep2_0(out)
        res2_1=self.mem_rep2_1(out)
        res.append(res2_0)
        res.append(res2_1)
        
        out=self.en_doubleConv4(torch.cat([res2_0['output'],res2_1['output']], dim=1))
        
        out=self.up1(out)#512
        out=self.de_doubleConv1(torch.cat([res1_0['output'],res1_1['output'],res1_2['output'],out], dim=1))
                               
        out=self.up2(out)
        out=self.de_doubleConv2(torch.cat([res0_0['output'],res0_1['output'],res0_2['output'],out], dim=1))
                                
        out=self.up3(out)
        out=self.de_doubleConv3(torch.cat([res_0['output'],res_1['output'],res_2['output'],out], dim=1)) 
                                
        out=self.up4(out)
        out=self.de_doubleConv4(out) 
        x=self.pyramid4_D(out)
                                
        return x,res


class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    def forward(self, x):
        gen_imag,res= self.encoder1(x)
        return gen_imag,res
    
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
            nn.Conv2d(ngf << 4, ngf << 5, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf << 5), 
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf << 5, 100, 4, 1, 0, bias=False),       # 512
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
    
