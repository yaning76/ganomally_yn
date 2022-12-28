import torch.nn as nn
import torch
from torchsummary import summary
# from transformer import Block




import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)
        self.norm = nn.LayerNorm(1024)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x,H,W):
 #         B, C, H, W = x.shape

#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         print(x.shape)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         print(x.shape)

        return x



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

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

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

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Block_Se(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1,
                 padding_mode='zeros', avg_pool=True,
                 se_block=True, activation=nn.SiLU(),
                 ):

        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.padding_mode = padding_mode
        self.se_block = se_block
        assert padding == 1
        padding_11 = padding - 3 // 2
        self.fused = False

        self.dense_groups = groups
        self.nonlinearity = activation

        self.rbr_identity = nn.BatchNorm2d(in_channels)
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=self.dense_groups) if (kernel_size != 1) else None
        self.rbr_1x1 = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups)
        if stride == 2 and avg_pool:
            self.rbr_1x1 = nn.Sequential(
                nn.AvgPool2d(2, 2),
                conv_bn(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0, groups=groups)
            )

        # updated to reuse code
        self.channel_shuffle = (groups > 1)

        if self.se_block:
            self.se = SE1(
                in_channels, out_channels, g=groups,
                ver=2 if (out_channels != in_channels or stride != 1) else 1)

    def _forward(self, inputs):
        if not self.fused:#1*1 conv
            rbr_1x1_output = self.rbr_1x1(inputs)
        else:
            rbr_1x1_output = None

        if self.rbr_dense is None:#3*3 conv
            dense_output = 0
        else:
            dense_output = self.rbr_dense(inputs)

        return rbr_1x1_output, dense_output

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)#BN

        rbr_1x1_output, drop_path_output = self._forward(inputs)

        if self.se_block:
            if self.rbr_identity is not None:
                id_out = id_out * self.se(id_out)###########

        if not self.fused:
            out = drop_path_output + rbr_1x1_output + id_out
        else:
            out = drop_path_output + id_out

        if self.se_block and (self.rbr_identity is None):
            out = out * self.se(inputs)############

        out = self.nonlinearity(out)
        return out
    
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize, nz, nc, ngf, ngpu, n_extra_layers=0, add_final_conv=True):
       
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        
        self.initial0 = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.pyramid0_ = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid0 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid1 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         self.stream=Block_Se(in_channels=512, out_channels=512)
        stream = nn.ModuleList()
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        self.stream=nn.Sequential(*stream)
        
        
        self.pyramid3 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if add_final_conv:
            self.final_conv=nn.Conv2d(1024, nz, 4, 1, 0, bias=False)
            
    def forward(self, input):
        output=self.initial0(input)
        output=self.pyramid0_(output)
        
        output=self.pyramid0(output)
        output=self.pyramid1(output)
        output=self.pyramid2(output)
#         print(output.shape)
        output = self.stream(output)
        output=self.pyramid3(output)
#         x.append(output)
        y=self.final_conv(output)    
        return output,y
    
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, imageSize, nz, nc, ngf, ngpu, n_extra_layers=0):
                
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        cngf, timageSize = ngf // 2, 4
        while timageSize != imageSize:
            cngf = cngf * 2
            timageSize = timageSize * 2
#         self.pyramid0_ = nn.Sequential(
#             nn.ConvTranspose2d(2048,1024, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(True),
#         )
        self.pyramid0 = nn.Sequential(
            nn.ConvTranspose2d(1024,512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.pyramid1 = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.pyramid2 = nn.Sequential(
            nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.pyramid3 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.pyramid4 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(32,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
#         input=self.pyramid0_(input)
        
        input=self.pyramid0(input)
        input=self.pyramid1(input)
        input=self.pyramid2(input)
        input=self.pyramid3(input)
        input=self.pyramid4(input)
        input=self.final0(input)
        return input
    
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt,dim=1024, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=5):
        super(NetG, self).__init__()
                
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
        self.transformer_blocks = nn.ModuleList([Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                linear=linear)
                for j in range(depths)])
        
        self.encoder1 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.decoder = Decoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
        self.encoder2 = Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    def forward(self, x):
        latent_i,y= self.encoder1(x)
        
        b,c,h,w=latent_i.shape
        latent_i = latent_i.flatten(2).transpose(1, 2)
        for i in range(len(self.transformer_blocks)):
            latent_i=self.transformer_blocks[i](latent_i,h,w)
        latent_i = latent_i.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
#         print(latent_i.shape)

        gen_imag = self.decoder(latent_i)
        _,latent_o = self.encoder2(gen_imag)
        
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag, y, latent_o
    
class NetD(nn.Module):
    
    def __init__(self,opt):
        super(NetD, self).__init__()
        
        self.initial0 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.pyramid0_ = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid0 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid1 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pyramid2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         self.stream=Block_Se(in_channels=512, out_channels=512)
        stream = nn.ModuleList()
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        stream.append(Block_Se(in_channels=512, out_channels=512))
        self.stream=nn.Sequential(*stream)
        
        
        self.pyramid3 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         self.pyramid4 = nn.Sequential(
#             nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(2048),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        self.final_conv=nn.Sequential(nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
                                nn.Sigmoid())
            
    def forward(self, input):
        output=self.initial0(input)
        output=self.pyramid0_(output)
        
        output=self.pyramid0(output)
        output=self.pyramid1(output)
        output=self.pyramid2(output)
#         print(output.shape)
        output = self.stream(output)
        output=self.pyramid3(output)
#         output=self.pyramid4(output)
#         x.append(output)
        y=self.final_conv(output)
        classifier = y.view(-1, 1).squeeze(1)
        return classifier,output