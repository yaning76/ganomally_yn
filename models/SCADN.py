import math
# from models.basic_modules import *
import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule,MemModule1,MemModule_w

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

    def __init__(self, imageSize=128, nc=3, ngf=64, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder, self).__init__()
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
        base_channels=64
        
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=2, padding=1),#128
            double_conv(base_channels*2, base_channels*2),
            nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, stride=2, padding=1),#64
            double_conv(base_channels*4, base_channels*4),
            nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3, stride=2, padding=1),#32
            double_conv(base_channels*8, base_channels*8),

#                     down(nc, base_channels),#64,128,128
#                     down(base_channels, base_channels*2),#128,64,64
#                     down(base_channels*2, base_channels*4),#256,32,32
            
            nn.Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, 
                      kernel_size=3, stride = 1,padding = 2,dilation = 2),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, 
                      kernel_size=3, stride = 1,padding = 2,dilation = 2),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, 
                      kernel_size=3, stride = 1,padding = 2,dilation = 2),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_channels*8, out_channels=base_channels*8, 
                      kernel_size=3, stride = 1,padding = 2,dilation = 2),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*8, base_channels*4, 1), 
#             nn.ConvTranspose2d(in_channels=base_channels*8, out_channels=base_channels*4, 
#                                kernel_size=3, stride=2, padding=1,output_padding=1),#256,64
            double_conv(base_channels*4, base_channels*4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*4, base_channels*2, 1), 
#             nn.ConvTranspose2d(in_channels=base_channels*4, out_channels=base_channels*2, 
#                                kernel_size=3, stride=2, padding=1,output_padding=1),#128,128
            double_conv(base_channels*2, base_channels*2),)
#             nn.ConvTranspose2d(in_channels=base_channels*2, out_channels=base_channels*2, 
#                                kernel_size=3, stride=2, padding=1,output_padding=1),#128,128
#             double_conv(base_channels*2, base_channels*2),)
#                     up(base_channels*4, base_channels*2),#128,64,64
#                     up(base_channels*2, base_channels*2),)#128,128,128


        
        self.initial0 = double_conv(nc, base_channels)#64,256,256
        self.pyramid0 = down(base_channels, base_channels*2)#128,128,128
        self.conv_2 = nn.Sequential(
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, 
                              kernel_size=3, stride = 1,padding = 2,dilation = 2),
                    nn.BatchNorm2d(base_channels*2),
                    nn.ReLU(inplace=True),
        )
        self.pyramid1_D = double_conv(base_channels*2, base_channels*2)#128,128,128
        self.pyramid2_D = up(base_channels*2, base_channels,bilinear=True)##64,256,256
#         self.pyramid2_D = up(base_channels*2, base_channels)##64,256,256
        self.final3_D =nn.Conv2d(base_channels, nc, 1)
        
    def forward(self, input):
#Encoder
        x=self.initial0(input)
        y=self.conv_1(x)
        x=self.pyramid0(x)
        x=x+y
        x=self.conv_2(x)
        x=self.pyramid1_D(x)
        x=self.pyramid2_D(x)
        x=self.final3_D(x)
        return x
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize,opt.nc, opt.ngf)
    def forward(self, x):
        gen_imag= self.encoder1(x)
        return gen_imag 
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
            nn.BatchNorm2d(ngf << 4),  
            # 1024
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
#         print("feature",feature.shape)
        classification = self.classify(feature)
#         print("classification",classification.shape)
        
        return classification.view(-1, 1).squeeze(1), feature    
    
