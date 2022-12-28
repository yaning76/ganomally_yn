import torch.nn as nn
import torch
from torchsummary import summary
# from .transformer import Block
# from .ACmix import ACmix

# from .SSE import Block_Se
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
            self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)

        self.add_final_conv=add_final_conv
        
        #SE
        
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        output=self.pyramid3(output)
        
        if self.add_final_conv:
            output=self.final_conv(output)
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
        ratio=16
        self.initial0 = nn.Sequential(#1,3,128,128->1,64,64,64
            nn.Conv2d(nc, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        self.final_conv_0_=nn.Conv2d(base_channels, nz, 4, 1, 0, bias=False)
        
#         self.final_conv_0_ = nn.Sequential(
#             nn.Linear(base_channels*64*64, base_channels*64*64//ratio),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(base_channels*64*64//ratio, nz)
#         )
        
        
        self.pyramid0 = nn.Sequential(#1,64,64,64->1,128,32,32
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         self.final_conv_0=nn.Sequential(
#             nn.Linear(base_channels*2*32*32, base_channels*2*32*32//ratio),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(base_channels*2*32*32//ratio, nz)
#         )
        self.pyramid1 = nn.Sequential(#1,128,32,32->1,256,16,16
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv_1=nn.Sequential(
            nn.Linear(base_channels*4*16*16, base_channels*4*16*16//ratio),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(base_channels*4*16*16//ratio, nz)
        )
        self.pyramid2 = nn.Sequential(#1,256,16,16->1,512,8,8
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv_2=nn.Sequential(
            nn.Linear(base_channels*8*8*8, base_channels*8*8*8//ratio),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(base_channels*8*8*8//ratio, nz)
        )
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if add_final_conv:
            self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)

        self.add_final_conv=add_final_conv
        
        #SE
        
        
    def forward(self, input):
        output=self.initial0(input)
#         z0_=self.final_conv_0_(output)
        output=self.pyramid0(output)
#         z0=self.final_conv_0(output)
        output=self.pyramid1(output)
        z1=torch.flatten(output, start_dim=1)
        z1=self.final_conv_1(z1)
        output=self.pyramid2(output)
        z2=torch.flatten(output, start_dim=1)
        z2=self.final_conv_2(z2)
        output=self.pyramid3(output)
        z3=self.final_conv(output)
        print(z1.shape,z2.shape, z3.shape)
        z = torch.cat([z1,z2, z3], 1)   
#         z=z1+z2+z3
        return z3,z
    
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
            nn.ConvTranspose2d(nz,base_channels*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(True),
        )
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
    
    def forward(self, input):
        
        out=self.initial0(input)
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
        latent_i= self.encoder1(x)
#         print('x,latent_i,y',x.shape,latent_i.shape,y.shape)
        gen_imag = self.decoder(latent_i)
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

