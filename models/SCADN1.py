import math
# from models.basic_modules import *
import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule,MemModule1,MemModule_w


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
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
        
        
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=2, padding=1),#128
            double_conv(base_channels*2, base_channels*2),
            nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, stride=2, padding=1),#64
            double_conv(base_channels*4, base_channels*4),
            nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3, stride=2, padding=1),#32
            double_conv(base_channels*8, base_channels*8),)
        
        residual_blocks1=4
        blocks1 = []
        for _ in range(residual_blocks1):
            block = ResnetBlock(base_channels*8, 2, use_spectral_norm=True)
            blocks1.append(block)
        self.middle_1 = nn.Sequential(*blocks1)
         
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=base_channels*8, out_channels=base_channels*4, 
                               kernel_size=3, stride=2, padding=1,output_padding=1),#256,64
            double_conv(base_channels*4, base_channels*4),
           
            nn.ConvTranspose2d(in_channels=base_channels*4, out_channels=base_channels*2, 
                               kernel_size=3, stride=2, padding=1,output_padding=1),#128,128
            double_conv(base_channels*2, base_channels*2),)
        
        self.initial0 = double_conv(nc, base_channels)#64,256,256
        self.initial1 = double_conv(nc, base_channels)#64,256,256
        
        self.encoder_2 = nn.Sequential(  
            down(base_channels, base_channels*2))#128,128,128
        residual_blocks2=4
        blocks2 = []
        for _ in range(residual_blocks2):
            block = ResnetBlock(base_channels*2, 2, use_spectral_norm=True)
            blocks2.append(block)
        self.middle_2 = nn.Sequential(*blocks2)

        self.decoder_2 = nn.Sequential(
            double_conv(base_channels*2, base_channels*2),#128,128,128
            up(base_channels*2, base_channels,bilinear=True),##64,256,256
#         self.pyramid2_D = up(base_channels*2, base_channels)##64,256,256
            nn.Conv2d(base_channels, nc, 1))
        self.decoder_1_ = nn.Sequential(
            double_conv(base_channels*2, base_channels*2),#128,128,128
            up(base_channels*2, base_channels,bilinear=True),##64,256,256
#         self.pyramid2_D = up(base_channels*2, base_channels)##64,256,256
            nn.Conv2d(base_channels, nc, 1))
        
    def forward(self, input1,input2):
#Encoder
        x=self.initial0(input1)
        y=self.initial1(input2)
        y=self.encoder_1(x)
        y=self.middle_1(y)
        y=self.decoder_1(y)
        y_=self.decoder_1_(y)
        
        x=self.encoder_2(x)
        x=x+y
#         print(x.shape)
#         print(y.shape)
        
        x=self.middle_2(x)
        x=self.decoder_2(x)
        return x,y_
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize,opt.nc, opt.ngf)
    def forward(self, x,x1):
        gen_imag,y= self.encoder1(x,x1)
        return gen_imag ,y
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
    
