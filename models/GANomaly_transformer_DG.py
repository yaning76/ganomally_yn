import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchstat import stat

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
#         self.norm = nn.LayerNorm(1024)

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
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        
        return x



class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, imageSize, nz, nc, ngf, ngpu, n_extra_layers=0):
        # nz : dimensionality of the latent space潜在空间的维度
        # nc : number of image channels
        # ndf : channels of middle layers for generator 生成器中间层通道数
        # ngpu : number of gpu
        # n_extra_layers : extra layers of Encoder and Decoder
        
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        cngf, timageSize = ngf // 2, 4
        while timageSize != imageSize:
            cngf = cngf * 2
            timageSize = timageSize * 2
        
        self.up1=Up(in_channels=2048, out_channels=1024, bilinear=True)
        self.up2=Up(in_channels=1024, out_channels=512, bilinear=True)
        self.up3=Up(in_channels=512, out_channels=256, bilinear=True)
        self.up4=Up(in_channels=256, out_channels=128, bilinear=True)
        self.up5=Up(in_channels=128, out_channels=64, bilinear=True)
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(64,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ) 
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x=self.final0(x)
        

        return x
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
        
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:

            return self.conv(x)
        
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize, nz, nc, ngf, ngpu, n_extra_layers=0, add_final_conv=True):
        # nz : dimensionality of the latent space潜在空间的维度
        # nc : number of image channels
        # ndf : channels of middle layers for generator 生成器中间层通道数
        # ngpu : number of gpu
        # n_extra_layers : extra layers of Encoder and Decoder
        
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"
        
        self.conv1=conv_3x3_bn(inp=3, oup=64, stride=2)
        self.Fused_MBConv_0=MBConv(inp=64, oup=64, stride=1, expand_ratio=1, use_se=0)
        self.Fused_MBConv_1=MBConv(inp=64, oup=128, stride=2, expand_ratio=4, use_se=0)
        self.Fused_MBConv_2=MBConv(inp=128, oup=256, stride=2, expand_ratio=4, use_se=0)
        
        self.MBConv_3=MBConv(inp=256, oup=512, stride=2, expand_ratio=4, use_se=1)
        self.MBConv_4=MBConv(inp=512, oup=1024, stride=2, expand_ratio=6, use_se=1)
        self.MBConv_5=MBConv(inp=1024, oup=2048, stride=2, expand_ratio=6, use_se=1)
        self.final=nn.Conv2d(2048, nz, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1(x)#[1, 64, 64, 64]
#         print(x.shape)
        x = self.Fused_MBConv_0(x)#[1, 64, 64, 64]
#         print(x.shape)
        x = self.Fused_MBConv_1(x)#1, 128, 32, 32
#         print(x.shape)
        x=self.Fused_MBConv_2(x)#1, 256, 16, 16
#         print(x.shape)
        x = self.MBConv_3(x)#1, 256, 16, 16
#         print(x.shape)
        x = self.MBConv_4(x)
#         print(x.shape)
        x = self.MBConv_5(x)#1, 512, 8, 8
#         print(x.shape)
        y = self.final(x)
        
        return x,y
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    

    def __init__(self,opt,dim=2048, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=5):
        super(NetG, self).__init__()
                
        imageSize=256
        nz=100
        nc=3
        ngf=64
        ngpu=1
        n_extra_layers=0
            
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        
        self.transformer_blocks = nn.ModuleList([Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                linear=linear)
                for j in range(depths)])
        
        self.encoder1 = Encoder(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        self.decoder = Decoder(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        self.encoder2 = Encoder(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    def forward(self, x):
        latent_i,y= self.encoder1(x)
        
        b,c,h,w=latent_i.shape
        latent_i = latent_i.flatten(2).transpose(1, 2)
        for i in range(len(self.transformer_blocks)):
            latent_i=self.transformer_blocks[i](latent_i,h,w)
        latent_i = latent_i.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
#         print(latent_i.shape)

        gen_imag = self.decoder(latent_i)
        latent_o,y1 = self.encoder2(gen_imag)
        
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag, y, y1

class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,opt):
        super(NetD, self).__init__()
        
        self.conv1=conv_3x3_bn(inp=3, oup=64, stride=2)
        self.Fused_MBConv_0=MBConv(inp=64, oup=64, stride=1, expand_ratio=1, use_se=0)
        self.Fused_MBConv_1=MBConv(inp=64, oup=128, stride=2, expand_ratio=4, use_se=0)
        self.Fused_MBConv_2=MBConv(inp=128, oup=256, stride=2, expand_ratio=4, use_se=0)
        
        self.MBConv_3=MBConv(inp=256, oup=512, stride=2, expand_ratio=4, use_se=1)
        self.MBConv_4=MBConv(inp=512, oup=1024, stride=2, expand_ratio=6, use_se=1)
        self.MBConv_5=MBConv(inp=1024, oup=2048, stride=2, expand_ratio=6, use_se=1)
        self.final=nn.Sequential(nn.Conv2d(2048, 1, 4, 1, 0, bias=False),
                                 nn.Sigmoid())
        
    def forward(self, x):
        x = self.conv1(x)#[1, 64, 64, 64]
#         print(x.shape)
        x = self.Fused_MBConv_0(x)#[1, 64, 64, 64]
#         print(x.shape)
        x = self.Fused_MBConv_1(x)#1, 128, 32, 32
#         print(x.shape)
        x=self.Fused_MBConv_2(x)#1, 256, 16, 16
#         print(x.shape)
        x = self.MBConv_3(x)#1, 256, 16, 16
#         print(x.shape)
        x = self.MBConv_4(x)
        features = self.MBConv_5(x)#1, 512, 8, 8
#         print(features.shape)
        classifier = self.final(features)#1, 1024, 4, 4
#         print(classifier.shape)
        classifier = classifier.view(-1, 1).squeeze(1)#[1, 1, 1, 1]

        return classifier, features
