import torch.nn as nn
import torch
from torchsummary import summary
# from .transformer import Block
# from .ACmix import ACmix
from .memory_module import MemModule,MemModule1,MemModule_w
# from .SSE import Block_Se


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
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        output=self.pyramid3(output)
        
        res_mem3= self.mem_rep3(output)
        f3 = res_mem3['output']
        att3 = res_mem3['att']
        out=self.pyramid0_D(f3)
        out=self.pyramid1_D(out)
        out=self.pyramid2_D(out)
        out=self.pyramid3_D(out)
        out=self.final0_D(out)
        
        return out,att3
    
# class Decoder(nn.Module):
#     """
#     DCGAN DECODER NETWORK
#     """
#     def __init__(self, imageSize=128, nz=1000, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
                
#         super(Decoder, self).__init__()
#         self.ngpu = ngpu
#         assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

#         cngf, timageSize = ngf // 2, 4
#         while timageSize != imageSize:
#             cngf = cngf * 2
#             timageSize = timageSize * 2
#         base_channels=64
#         self.initial0 = nn.Sequential(
#             nn.ConvTranspose2d(nz,base_channels*16, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(base_channels*16),
#             nn.ReLU(True),
#         )
#         self.pyramid0 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels*16,base_channels*8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(base_channels*8),
#             nn.ReLU(True),
#         )
        
#         self.pyramid1 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels*8,base_channels*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(base_channels*4),
#             nn.ReLU(True),
#         )
#         self.pyramid2 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels*4,base_channels*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(base_channels*2),
#             nn.ReLU(True),
#         )
#         self.pyramid3 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels*2,base_channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(True),
#         )
#         self.final0 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels,nc, 4, 2, 1, bias=False),
#             nn.Tanh(),
#         )
    
#     def forward(self, input):
        
#         out=self.initial0(input)
#         out=self.pyramid0(out)
#         out=self.pyramid1(out)
#         out=self.pyramid2(out)
#         out=self.pyramid3(out)
#         out=self.final0(out)
#         return out
    
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#         self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#         self.encoder2 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    def forward(self, x):
        gen_imag,att3= self.encoder1(x)
        return gen_imag,att3  
        


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
    
