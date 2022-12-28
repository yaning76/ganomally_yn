import torch.nn as nn
import torch
from torchsummary import summary
from .memory_module import MemModule
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
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
        self.mem_rep2 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*8, shrink_thres =shrink_thres)
        self.DoubleConv2=DoubleConv(in_channels=base_channels*16, out_channels=base_channels*8)
        
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
        self.mem_rep1 = MemModule(mem_dim=mem_dim, fea_dim=base_channels*4, shrink_thres =shrink_thres)
        self.DoubleConv1=DoubleConv(in_channels=base_channels*8, out_channels=base_channels*4)
        
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
        output=self.pyramid0(output)
        output_1=self.pyramid1(output)
        output_2=self.pyramid2(output_1)
        
        
        
        output=self.pyramid3(output_2)
        res_mem3= self.mem_rep3(output)
        f3 = res_mem3['output']
        att3 = res_mem3['att']

#Decoder
        out=self.pyramid0_D(f3)
        out_2 = torch.cat([output_2, out], 1)
        out=self.DoubleConv2(out_2)
#         print(out.shape)
        res_mem2 = self.mem_rep2(out)
        f2 = res_mem2['output']
        att2 = res_mem2['att']
        out=self.pyramid1_D(f2)
        out_1 = torch.cat([output_1, out], 1)
        out=self.DoubleConv1(out_1)
#         print(out.shape)
        res_mem1 = self.mem_rep1(out)
        f1 = res_mem1['output']
        att1 = res_mem1['att']
        out=self.pyramid2_D(f1)
        out=self.pyramid3_D(out)
        out=self.final0_D(out)
        return out,att1,att2,att3
    
    
    
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
#         self.initial0 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels*16,base_channels*16, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(base_channels*16),
#             nn.ReLU(True),
#         )
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
        
#         out=self.initial0(input)
        out=self.pyramid0(input)
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
    def forward(self, x):
        gen_imag,att1,att2,att3= self.encoder1(x)
        return gen_imag,att1,att2,att3
      
# class NetG(nn.Module):
#     """
#     GENERATOR NETWORK
#     """

#     def __init__(self,opt):
#         super(NetG, self).__init__()
# #         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
#         self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#         self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

#     def forward(self, x):
#         f,att2,att3= self.encoder1(x)
#         gen_imag = self.decoder(f)
#         return gen_imag,att2,att3
        

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
# class NetD(nn.Module):
    
#     def __init__(self,opt):
#         super(NetD, self).__init__()
#         base_channels=64
#         self.encoder1 = Encoder_D(add_final_conv=False)
#         self.final_conv=nn.Sequential(nn.Conv2d(base_channels*16, 1, 4, 1, 0, bias=False),
#                                 nn.Sigmoid())
            
#     def forward(self, input):
# #         print(input.shape)
#         output=self.encoder1(input)
# #         print(output.shape)
        
#         y=self.final_conv(output)
#         classifier = y.view(-1, 1).squeeze(1)
#         return classifier,output

