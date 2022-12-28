import torch.nn as nn
import torch
from torchsummary import summary
from .transformer import Block
from .ACmix import ACmix

# from .SSE import Block_Se

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#         print(p)
    return p
class SE1(nn.Module):
    # Squeeze-and-excitation block in https://arxiv.org/abs/1709.01507
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c_in, c_out, n=1, shortcut=True,  g=1, e=0.5, ver=1):
        super(SE1, self).__init__()
        self.ver = ver
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cvsig = ConvSig(c_in, c_out, 1, 1, g=g)

    def forward(self, x):
        x = self.cvsig(self.avg_pool(x))
        if self.ver == 2:
            x = 2 * x
        return x
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
class ConvSig(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid() if act else nn.Identity()
#         print(autopad(k, p))
    def forward(self, x):
        
        return self.act(self.conv(x))

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, dilation=1, groups=1, avg_pool=True,
                 se_block=True, activation=nn.ReLU(),
                 ):

        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.se_block = se_block

        assert padding == 1
        padding_11 = padding - 3 // 2

        self.fused = False

        self.dense_groups = groups
        self.nonlinearity = activation
        
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=1) if (kernel_size != 1) else None
        
        self.rbr_1x1 = nn.Sequential(
                nn.AvgPool2d(2, 2),
                conv_bn(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0, groups=groups))

        if self.se_block:
            self.se = SE1(
                in_channels, out_channels, g=groups,
                ver=2 if (out_channels != in_channels or stride != 1) else 1)

    def forward(self, inputs):
        rbr_1x1_output=self.rbr_1x1(inputs)
        drop_path_output = self.rbr_dense(inputs)
        out = drop_path_output + rbr_1x1_output 
        out = out * self.se(inputs)############
        out = self.nonlinearity(out)
        return out

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
        
        self.acmix1=ACmix(in_planes=512, out_planes=512, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
        
       
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.acmix2=ACmix(in_planes=1024, out_planes=1024, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
#         if add_final_conv:
        self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)
#         self.add_final_conv=add_final_conv    
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
#         print(output.shape)
        output=self.acmix1(output)
#         print(output.shape)
        output=self.pyramid3(output)
        output=self.acmix2(output)
#         x.append(output)
#         if self.add_final_conv:
        y=self.final_conv(output)   
#             return y
#         else:
#             return output
        return output,y
class Encoder_1(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize=128, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder_1, self).__init__()
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
        
        self.acmix1=ACmix(in_planes=512, out_planes=512, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
        
       
        self.pyramid3 = nn.Sequential(#1,512,8,8->1,1024,4,4
            nn.Conv2d(base_channels*8, base_channels*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.acmix2=ACmix(in_planes=1024, out_planes=1024, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
#         if add_final_conv:
        self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)
#         self.add_final_conv=add_final_conv    
    def forward(self, input):
        output=self.initial0(input)
        
        output=self.pyramid0(output)
        
        output=self.pyramid1(output)
        
        output=self.pyramid2(output)
#         print(output.shape)
        output=self.acmix1(output)
#         print(output.shape)
        output=self.pyramid3(output)
        output=self.acmix2(output)
#         x.append(output)
#         if self.add_final_conv:
#             return y
#         else:
#             return output
        return output

# class Encoder_(nn.Module):
#     """
#     DCGAN ENCODER NETWORK
#     """

#     def __init__(self, imageSize=128, nz=1000, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
       
#         super(Encoder_, self).__init__()
#         self.ngpu = ngpu
#         assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
#         base_channels=64
        
#         self.initial0=DownsamplingBlock(in_channels=3, out_channels=base_channels, kernel_size=3)
#         self.pyramid0=DownsamplingBlock(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3)
#         self.pyramid1=DownsamplingBlock(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3)
#         self.pyramid2=DownsamplingBlock(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3)
#         self.pyramid3=DownsamplingBlock(in_channels=base_channels*8, out_channels=base_channels*16, kernel_size=3)
     
#         if add_final_conv:
#             self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)
#         self.add_final_conv=add_final_conv    
#     def forward(self, input):
#         output=self.initial0(input)
        
#         output=self.pyramid0(output)
        
#         output=self.pyramid1(output)
        
#         output=self.pyramid2(output)
        
# #         print(output.shape)
#         output=self.pyramid3(output)
# #         x.append(output)
#         if self.add_final_conv:
#             y=self.final_conv(output)   
#             return y
#         else:
#             return output
        
# class Encoder(nn.Module):
#     """
#     DCGAN ENCODER NETWORK
#     """
#     def __init__(self, imageSize=128, nz=1000, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True,
#                  dim=1024, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=4):
       
#         super(Encoder, self).__init__()
#         self.ngpu = ngpu
#         assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
#         base_channels=64
        
#         self.initial0=DownsamplingBlock(in_channels=3, out_channels=base_channels, kernel_size=3)
#         self.pyramid0=DownsamplingBlock(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3)
#         self.pyramid1=DownsamplingBlock(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3)
#         self.pyramid2=DownsamplingBlock(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3)
#         self.pyramid3=DownsamplingBlock(in_channels=base_channels*8, out_channels=base_channels*16, kernel_size=3)
        
#         self.depths = depths
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
#         self.transformer_blocks = nn.ModuleList([Block(
#                 dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
#                 linear=linear)
#                 for j in range(depths)])
        
        
#         if add_final_conv:
#             self.final_conv=nn.Conv2d(base_channels*16, nz, 4, 1, 0, bias=False)
        
#     def forward(self, input):
#         out=self.initial0(input)
#         out=self.pyramid0(out)    
#         out=self.pyramid1(out)
#         out_=self.pyramid2(out)
        
#         out=self.pyramid3(out_)
        
#         b,c,h,w=out.shape
#         out1 = out.flatten(2).transpose(1, 2)
#         for i in range(len(self.transformer_blocks)):
#             out1=self.transformer_blocks[i](out1,h,w)
#         out1 = out1.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        
#         y=self.final_conv(out)
# #         print('out_,out1,y',out_.shape,out1.shape,y.shape)
#         return out_,out1,y
    
    
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
        self.pyramid0_ = nn.Sequential(
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
#         self.pyramid4 = nn.Sequential(
#             nn.ConvTranspose2d(64,32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#         )
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(base_channels,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
#         self.down3 = nn.Sequential(
#             nn.Conv2d(base_channels*16, base_channels*8, 1, 1,bias=False),
#             nn.BatchNorm2d(base_channels*8),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

    def forward(self, input):
        
        out=self.pyramid0_(input)
#         print('out',out.shape)
        out=self.pyramid0(out)
#         print('out',out.shape)
        
#         out_= torch.cat([out, x], 1)
#         out=self.down3(out_)
        
        out=self.pyramid1(out)
        out=self.pyramid2(out)
        out=self.pyramid3(out)
#         input=self.pyramid4(input)
        out=self.final0(out)
        return out
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt):
        super(NetG, self).__init__()
#         opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers
        self.encoder1 = Encoder_(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.encoder2 = Encoder_(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    def forward(self, x):
        y,latent_i= self.encoder1(x)
#         print('x,latent_i,y',x.shape,latent_i.shape,y.shape)
        gen_imag = self.decoder(latent_i)
#         print('gen_imag',gen_imag.shape)
        y_,latent_o = self.encoder2(gen_imag)
        
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag,latent_i, latent_o
        


class NetD(nn.Module):
    
    def __init__(self,opt):
        super(NetD, self).__init__()
        base_channels=64
        self.encoder1 = Encoder_1(add_final_conv=False)
        self.final_conv=nn.Sequential(nn.Conv2d(base_channels*16, 1, 4, 1, 0, bias=False),
                                nn.Sigmoid())
            
    def forward(self, input):
        output=self.encoder1(input)
        y=self.final_conv(output)
        classifier = y.view(-1, 1).squeeze(1)
        return classifier,output

