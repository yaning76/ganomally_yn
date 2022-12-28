import torch.nn as nn
import torch
from torchsummary import summary
# from .transformer import Block
# from .ACmix import ACmix

# from .SSE import Block_Se
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
class Encoder_(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder_, self).__init__()
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
            self.final_conv0=nn.Conv2d(base_channels*16, 512, 4, 1, 0, bias=False)
            

        self.add_final_conv=add_final_conv
        
        
        
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        output=self.pyramid3(output)
        
        if self.add_final_conv:
            output=self.final_conv0(output) 
        return output
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder, self).__init__()
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
            self.final_conv0=nn.Conv2d(base_channels*16, 512, 4, 1, 0, bias=False)
            self.final_conv1=nn.Conv2d(base_channels*16, 1024, 4, 1, 0, bias=False)
            

        self.add_final_conv=add_final_conv
        
        
        
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        output=self.pyramid3(output)
        
        if self.add_final_conv:
            output0=self.final_conv0(output) 
            output1=self.final_conv1(output) 
        return output0,output1
    
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, imageSize=128, nz=1000, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
                
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        cngf, timageSize = ngf // 2, 4
        while timageSize != imageSize:
            cngf = cngf * 2
            timageSize = timageSize * 2
        base_channels=64
        self.initial0 = nn.Sequential(
            nn.ConvTranspose2d(512,base_channels*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(True),
        )
        self.initial0_ = nn.Sequential(
            nn.ConvTranspose2d(1024,base_channels*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(True),
        )
        self.conv=BaseConv(2048,1024,3,1)
        self.pyramid0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*16,base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True),
        )
        
        self.pyramid1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8,base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),
        )
        self.pyramid2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4,base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),
        )
        self.pyramid3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2,base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
        )
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, input1,input2):
        
        out0=self.initial0(input1)
        out1=self.initial0_(input2)
        out = torch.cat([out0, out1], 1) 
        out=self.conv(out)
        out=self.pyramid0(out)
        out=self.pyramid1(out)
        out=self.pyramid2(out)
        out=self.pyramid3(out)
        out=self.final0(out)
        return out
    
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.encoder2 = Encoder_(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    def forward(self, x):
        latent_i,y= self.encoder1(x)
#         print('x,latent_i,y',x.shape,latent_i.shape,y.shape)
        gen_imag = self.decoder(latent_i,y)
#         print('gen_imag',gen_imag.shape)
        latent_o = self.encoder2(gen_imag)
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag,latent_i, latent_o
        


class NetD(nn.Module):
    
    def __init__(self,opt):
        super(NetD, self).__init__()
        base_channels=64
        self.encoder1 = Encoder_(add_final_conv=False)
        self.final_conv=nn.Sequential(nn.Conv2d(base_channels*16, 1, 4, 1, 0, bias=False),
                                nn.Sigmoid())
            
    def forward(self, input):
        output=self.encoder1(input)
        y=self.final_conv(output)
        classifier = y.view(-1, 1).squeeze(1)
        return classifier,output

