import math
# from models.basic_modules import *
import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule,MemModule1,MemModule_w
import torch.nn.functional as F

# Squeeze and Excitation block
class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# SSPCAB implementation
class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)#维度填充

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x
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
        window_size=2
        
        self.initial0 = double_conv(nc, base_channels)#128
        self.pyramid0 = down(base_channels, base_channels*2)#64
        self.pyramid1 = down(base_channels*2, base_channels*4)#32
        self.SSPCAB1=SSPCAB(channels=base_channels*4)
        self.pyramid2 = down(base_channels*4, base_channels*8)#16
        self.SSPCAB2=SSPCAB(channels=base_channels*8)
        self.pyramid3 = down(base_channels*8, base_channels*16)#8
#         self.SSPCAB3=SSPCAB(channels=base_channels*16)
        
        self.mem_rep3 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*16, shrink_thres =shrink_thres)
        self.mem_rep3_ = MemModule1(mem_dim=mem_dim, fea_dim=64, shrink_thres =shrink_thres)
        self.conv3 = double_conv(base_channels*32,base_channels*16)
#         self.conv3 = double_conv(base_channels*32,base_channels*16)
        self.mem_rep2 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*8, shrink_thres =shrink_thres)
        self.mem_rep2_ = MemModule1(mem_dim=mem_dim, fea_dim=256, shrink_thres =shrink_thres)
        self.conv2 = double_conv(base_channels*24,base_channels*8)
        
        self.mem_rep1 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*4, shrink_thres =shrink_thres)
        self.mem_rep1_ = MemModule1(mem_dim=mem_dim, fea_dim=1024, shrink_thres =shrink_thres)
        self.conv1 = double_conv(base_channels*12,base_channels*4)
        
        self.mem_rep0 = MemModule_w(mem_dim=mem_dim, fea_dim=base_channels*2*window_size*window_size, 
                                    shrink_thres =shrink_thres)
        self.conv0 = double_conv(base_channels*4,base_channels*2)
#         self.mem_rep0_ = MemModule_w(mem_dim=mem_dim, fea_dim=base_channels*window_size*window_size, 
#                                     shrink_thres =shrink_thres)
#         self.conv0_ = double_conv(base_channels*2,base_channels*2)
        
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
        out1_=self.SSPCAB1(out1)
        out2=self.pyramid2(out1_)
        out2_=self.SSPCAB2(out2)
        out3=self.pyramid3(out2)
#         out3_=self.SSPCAB3(out3)
        
        res_mem3= self.mem_rep3(out3)
        f3 = res_mem3['output']
        att3 = res_mem3['att']
        res_mem3_= self.mem_rep3_(out3)
        f3_ = res_mem3_['output']
        att3_ = res_mem3_['att']
        
        res_mem2= self.mem_rep2(out2)
        f2 = res_mem2['output']
        att2 = res_mem2['att']
        res_mem2_= self.mem_rep2_(out2)
        f2_ = res_mem2_['output']
        att2_ = res_mem2_['att']
        
        res_mem1= self.mem_rep1(out1)
        f1 = res_mem1['output']
        att1 = res_mem1['att']
        res_mem1_= self.mem_rep1_(out1)
        f1_ = res_mem1_['output']
        att1_ = res_mem1_['att']
        
        res_mem0= self.mem_rep0(out0)
        f0 = res_mem0['output']
        att0 = res_mem0['att']
#         res_mem0_= self.mem_rep0_(out)
#         f0_ = res_mem0_['output']
#         att0_ = res_mem0_['att']
        
#Decoder
        x = torch.cat([f3, f3_], dim=1)
        x=self.conv3(x)
        x=self.pyramid0_D(f3)
        x = torch.cat([f2, f2_,x], dim=1)
        x=self.conv2(x)
        
        x=self.pyramid1_D(x)
        x = torch.cat([f1, f1_,x], dim=1)
        x=self.conv1(x)
        x=self.pyramid2_D(x)
        x = torch.cat([f0,x], dim=1)
        x=self.conv0(x)
        x=self.pyramid3_D(x)
#         x = torch.cat([x, f0_], dim=1)
#         x=self.conv0_(x)
        x=self.pyramid4_D(x)
        
#         print(x.shape)
        return x,out1_,out1,out2_,out2,att3,att3_,att2,att2_,att1,att1_,att0
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    def forward(self, x):
        gen_imag,out1_,out1,out2_,out2,att3,att3_,att2,att2_,att1,att1_,att0= self.encoder1(x)
        return gen_imag,out1_,out1,out2_,out2,att3,att3_,att2,att2_,att1,att1_,att0
    
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
    
