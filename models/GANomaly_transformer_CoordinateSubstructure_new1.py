import torch.nn as nn
import torch
from torchsummary import summary
from .transformer import Block
from .SSE import Block_Se

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
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
        
#         print(output.shape)
        output=self.pyramid3(output)
#         x.append(output)
        if self.add_final_conv:
            y=self.final_conv(output)   
            return y
        else:
            return output
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True,
                 dim=1024, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=4):
       
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
        stream0 = nn.ModuleList()
        stream0.append(Block_Se(in_channels=base_channels*2, out_channels=base_channels*2,activation=nn.ReLU(True)))#1,128,32,32->1,128,32,32
        stream0.append(Block_Se(in_channels=base_channels*2, out_channels=base_channels*2,activation=nn.ReLU(True)))#1,128,32,32->1,128,32,32
#         stream0.append(Block_Se(in_channels=base_channels*2, out_channels=base_channels*2))#1,128,32,32->1,128,32,32
        self.stream0=nn.Sequential(*stream0)
        self.down0 = Block_Se(in_channels=base_channels*2, out_channels=base_channels*4,kernel_size=3,stride=2)#1,128,32,32->1,256,16,16
        
        
        self.pyramid1 = nn.Sequential(#1,128,32,32->1,256,16,16
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        stream1 = nn.ModuleList()
        stream1.append(Block_Se(in_channels=base_channels*4, out_channels=base_channels*4,activation=nn.ReLU(True)))#1,256,16,16->1,256,16,16
        stream1.append(Block_Se(in_channels=base_channels*4, out_channels=base_channels*4,activation=nn.ReLU(True)))#1,256,16,16->1,256,16,16
        self.stream1=nn.Sequential(*stream1)
        self.down1 = Block_Se(in_channels=base_channels*8, out_channels=base_channels*8,kernel_size=3,stride=2)#1,512,16,16->1,512,8,8
        
        self.pyramid2 = nn.Sequential(#1,256,16,16->1,512,8,8
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        stream2 = nn.ModuleList()
        stream2.append(Block_Se(in_channels=base_channels*8, out_channels=base_channels*8,activation=nn.ReLU(True)))#1,512,8,8->1,512,8,8
        stream2.append(Block_Se(in_channels=base_channels*8, out_channels=base_channels*8,activation=nn.ReLU(True)))#1,512,8,8->1,512,8,8
        self.stream2=nn.Sequential(*stream2)
        self.down2 = Block_Se(in_channels=base_channels*16, out_channels=base_channels*16,kernel_size=3,stride=2)#1,1024,8,8->1,1024,4,4
        
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         self.down3 = Block_Se(in_channels=base_channels*32, out_channels=base_channels*16,kernel_size=1,stride=1)#1,2048,4,4->1,1024,4,4
        
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels*32, base_channels*16, 1, 1,bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
        self.transformer_blocks = nn.ModuleList([Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                linear=linear)
                for j in range(depths)])
        
        
        if add_final_conv:
            self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)
            
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        output0=self.stream0(output)
        output0=self.down0(output)
        
        output=self.pyramid1(output)
        output1=self.stream1(output)
        output_01 = torch.cat([output0, output1], 1)
        output1=self.down1(output_01)
        
        output=self.pyramid2(output)
        output2=self.stream2(output)
        output_12 = torch.cat([output1, output2], 1)
        output2=self.down2(output_12)
        
#         print(output.shape)
        output=self.pyramid3(output)
        output_23 = torch.cat([output2, output], 1)
        output2=self.down3(output_23)
        
        b,c,h,w=output2.shape
        output2 = output2.flatten(2).transpose(1, 2)
        for i in range(len(self.transformer_blocks)):
            output2=self.transformer_blocks[i](output2,h,w)
        output2 = output2.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        
        
#         x.append(output)
        y=self.final_conv(output)    
        return output,output2,y
    
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
                
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        cngf, timageSize = ngf // 2, 4
        while timageSize != imageSize:
            cngf = cngf * 2
            timageSize = timageSize * 2
        base_channels=64
        
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
#         self.pyramid4 = nn.Sequential(
#             nn.ConvTranspose2d(64,32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#         )
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.cat_down=nn.Conv2d(base_channels*32, base_channels*16, 1, 1, bias=False)
    def forward(self, input,encoder_input):
#         print(input.shape,encoder_input.shape)
        cat = torch.cat([encoder_input, input], 1)
#         print(input.shape,encoder_input.shape,cat.shape)
        input=self.cat_down(cat)
#         print(input.shape)
        input=self.pyramid0(input)
        input=self.pyramid1(input)
        input=self.pyramid2(input)
        input=self.pyramid3(input)
#         input=self.pyramid4(input)
        input=self.final0(input)
        return input
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
        output,latent_i,y= self.encoder1(x)
#         print('output,latent_i,y',output.shape,latent_i.shape,y.shape)
        
        gen_imag = self.decoder(latent_i,output)
        latent_o = self.encoder2(gen_imag)
        
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag, y, latent_o
        
# class NetG(nn.Module):
#     """
#     GENERATOR NETWORK
#     """

#     def __init__(self,opt,dim=1024, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=5):
#         super(NetG, self).__init__()
                
#         self.depths = depths
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
#         self.transformer_blocks = nn.ModuleList([Block(
#                 dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
#                 linear=linear)
#                 for j in range(depths)])
        
#         self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#         self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#         self.encoder2 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

#     def forward(self, x):
#         latent_i,y= self.encoder1(x)
        
#         b,c,h,w=latent_i.shape
#         latent_i = latent_i.flatten(2).transpose(1, 2)
#         for i in range(len(self.transformer_blocks)):
#             latent_i=self.transformer_blocks[i](latent_i,h,w)
#         latent_i = latent_i.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
# #         print(latent_i.shape)

#         gen_imag = self.decoder(latent_i)
#         _,latent_o = self.encoder2(gen_imag)
        
# #         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
#         return gen_imag, y, latent_o

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

# class NetD(nn.Module):
    
#     def __init__(self,opt):
#         super(NetD, self).__init__()
        
#         self.initial0 = nn.Sequential(
#             nn.Conv2d(3, 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True)
#         )
#         self.pyramid0_ = nn.Sequential(
#             nn.Conv2d(32, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.pyramid0 = nn.Sequential(
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.pyramid1 = nn.Sequential(
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.pyramid2 = nn.Sequential(
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
# #         self.stream=Block_Se(in_channels=512, out_channels=512)
#         stream = nn.ModuleList()
#         stream.append(Block_Se(in_channels=512, out_channels=512))
#         stream.append(Block_Se(in_channels=512, out_channels=512))
#         stream.append(Block_Se(in_channels=512, out_channels=512))
#         stream.append(Block_Se(in_channels=512, out_channels=512))
#         stream.append(Block_Se(in_channels=512, out_channels=512))
#         self.stream=nn.Sequential(*stream)
        
        
#         self.pyramid3 = nn.Sequential(
#             nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
# #         self.pyramid4 = nn.Sequential(
# #             nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(2048),
# #             nn.LeakyReLU(0.2, inplace=True),
# #         )
#         self.final_conv=nn.Sequential(nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
#                                 nn.Sigmoid())
            
#     def forward(self, input):
#         output=self.initial0(input)
#         output=self.pyramid0_(output)
        
#         output=self.pyramid0(output)
#         output=self.pyramid1(output)
#         output=self.pyramid2(output)
# #         print(output.shape)
#         output = self.stream(output)
#         output=self.pyramid3(output)
# #         output=self.pyramid4(output)
# #         x.append(output)
#         y=self.final_conv(output)
#         classifier = y.view(-1, 1).squeeze(1)
#         return classifier,output