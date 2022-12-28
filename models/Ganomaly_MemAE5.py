import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
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
        
        self.initial0 = nn.Sequential(#1,3,128,128->1,64,64,64
            nn.Conv2d(nc, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=base_channels, shrink_thres =shrink_thres)
        self.BaseConv_0=BaseConv(in_channels=base_channels*2, out_channels=base_channels*2, ksize=3, stride=1)
        self.BaseConv_1=BaseConv(in_channels=base_channels*2, out_channels=base_channels, ksize=1, stride=1)
        self.pyramid0 = nn.Sequential(#1,64,64,64->1,128,32,32
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mem_rep0 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*2, shrink_thres =shrink_thres)
        self.BaseConv0_0=BaseConv(in_channels=base_channels*4, out_channels=base_channels*4, ksize=3, stride=1)
        self.BaseConv0_1=BaseConv(in_channels=base_channels*4, out_channels=base_channels*2, ksize=1, stride=1)
        
        self.pyramid1 = nn.Sequential(#1,128,32,32->1,256,16,16
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mem_rep1 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*4, shrink_thres =shrink_thres)
        self.BaseConv1_0=BaseConv(in_channels=base_channels*8, out_channels=base_channels*8, ksize=3, stride=1)
        self.BaseConv1_1=BaseConv(in_channels=base_channels*8, out_channels=base_channels*4, ksize=1, stride=1)
        
        self.pyramid2 = nn.Sequential(#1,256,16,16->1,512,8,8
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mem_rep2 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*8, shrink_thres =shrink_thres)
        self.BaseConv2_0=BaseConv(in_channels=base_channels*16, out_channels=base_channels*16, ksize=3, stride=1)
        self.BaseConv2_1=BaseConv(in_channels=base_channels*16, out_channels=base_channels*8, ksize=1, stride=1)
        
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mem_rep3 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*16, shrink_thres =shrink_thres)
        
        
        self.pyramid0_D = nn.Sequential(
            nn.ConvTranspose2d(base_channels*16,base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True),
        )
        
        self.pyramid1_D = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8,base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),
        )
        self.pyramid2_D = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4,base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),
        )
        self.pyramid3_D = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2,base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
        )
        self.final0_D = nn.Sequential(
            nn.ConvTranspose2d(base_channels,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, input):
#Encoder
        output=self.initial0(input)
        res_mem = self.mem_rep(output)
        f = res_mem['output']
        att = res_mem['att']
        output=self.pyramid0(output)
        res_mem0 = self.mem_rep0(output)
        f0 = res_mem0['output']
        att0 = res_mem0['att']
        output=self.pyramid1(output)
        res_mem1 = self.mem_rep1(output)
        f1 = res_mem1['output']
        att1 = res_mem1['att']
        
        output=self.pyramid2(output)
        
        res_mem2 = self.mem_rep2(output)
        f2 = res_mem2['output']
        att2 = res_mem2['att']
        
        output=self.pyramid3(output)
        res_mem3= self.mem_rep3(output)
        f3 = res_mem3['output']
        att3 = res_mem3['att']
#         if self.add_final_conv:
#             output=self.final_conv(output) 

#Decoder
        out=self.pyramid0_D(f3)
#         print(out.shape)
    
        out_2 = torch.cat([f2, out], 1)
        out_2=self.BaseConv2_0(out_2)
        out=self.BaseConv2_1(out_2)
        
        out=self.pyramid1_D(out)
        out_1 = torch.cat([f1, out], 1)
        out_1=self.BaseConv1_0(out_1)
        out=self.BaseConv1_1(out_1)
        
        out=self.pyramid2_D(out)
        out_0 = torch.cat([f0, out], 1)
        out_0=self.BaseConv0_0(out_0)
        out=self.BaseConv0_1(out_0)
        
        out=self.pyramid3_D(out)
        out_ = torch.cat([f, out], 1)
        out_=self.BaseConv_0(out_)
        out=self.BaseConv_1(out_)
        out=self.final0_D(out)
        return out,att2,att3,att1,att0,att
    
    
    
class Encoder_D(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder_D, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
        base_channels=64
        
        self.initial0 = nn.Sequential(#1,3,128,128->1,64,64,64
            nn.Conv2d(nc, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        self.pyramid0 = nn.Sequential(#1,64,64,64->1,128,32,32
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.pyramid1 = nn.Sequential(#1,128,32,32->1,256,16,16
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
       
        self.pyramid2 = nn.Sequential(#1,256,16,16->1,512,8,8
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        if add_final_conv:
            self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)

        self.add_final_conv=add_final_conv
        
        
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        output=self.pyramid3(output)
        if self.add_final_conv:
            output=self.final_conv(output) 
        return output
        
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    def forward(self, x):
        gen_imag,att2,att3,att1,att0,att= self.encoder1(x)
        return gen_imag,att2,att3,att1,att0,att
      

class NetD(nn.Module):
    
    def __init__(self,opt):
        super(NetD, self).__init__()
        base_channels=64
        self.encoder1 = Encoder_D(add_final_conv=False)
        self.final_conv=nn.Sequential(nn.Conv2d(base_channels*16, 1, 4, 1, 0, bias=False),
                                nn.Sigmoid())
            
    def forward(self, input):
#         print(input.shape)
        output=self.encoder1(input)
#         print(output.shape)
        
        y=self.final_conv(output)
        classifier = y.view(-1, 1).squeeze(1)
        return classifier,output

