import torch.nn as nn
import torch
from torchsummary import summary
from transformer import Block

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

        
        self.initial0 = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        self.pyramid0 = nn.Sequential(
            nn.Conv2d(ngf, 128, 4, 2, 1, bias=False),
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
        self.pyramid3 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
#         if add_final_conv:
#             self.final_conv=nn.Conv2d(1024, nz, 4, 1, 0, bias=False)
            
    def forward(self, input):
        x=[]
        
        output=self.initial0(input)
        x.append(output)
        
        output=self.pyramid0(output)
        x.append(output)
        
        output=self.pyramid1(output)
        x.append(output)
        
        output=self.pyramid2(output)
        x.append(output)
        
        output=self.pyramid3(output)
#         x.append(output)
            
        return output,x

    
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
        self.final0 = nn.Sequential(
            nn.ConvTranspose2d(64,nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        
        self.double_conv0 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, input,y):
        num=len(y)-1
        
        input=self.pyramid0(input)
        input=torch.cat([y[num], input], dim=1)
        input=self.double_conv1(input)
        num=num-1
        
        input=self.pyramid1(input)
        input=torch.cat([y[num], input], dim=1)
        input=self.double_conv2(input)
        num=num-1
        
        input=self.pyramid2(input)
        input=torch.cat([y[num], input], dim=1)
        input=self.double_conv3(input)
        num=num-1
        
        input=self.pyramid3(input)
        input=torch.cat([y[num], input], dim=1)
        input=self.double_conv4(input)
        
        input=self.final0(input)
        
        return input
class Encoder_ori(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, imageSize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        # nz : dimensionality of the latent space潜在空间的维度
        # nc : number of image channels
        # ndf : channels of middle layers for generator 生成器中间层通道数
        # ngpu : number of gpu
        # n_extra_layers : extra layers of Encoder and Decoder
        
        super(Encoder_ori, self).__init__()
        self.ngpu = ngpu
        assert imageSize % 16 == 0, "imageSize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x imageSize x imageSize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = imageSize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
#         if add_final_conv:
#             main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
#                             nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

    
    
    
 
    
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self,opt,dim=1024, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, linear=False,depths=4):
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
        self.encoder2 = Encoder_ori(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    def forward(self, x):
        latent_i,out = self.encoder1(x)
        
        b,c,h,w=latent_i.shape
        latent_i = latent_i.flatten(2).transpose(1, 2)
        for i in range(len(self.transformer_blocks)):
            latent_i=self.transformer_blocks[i](latent_i,h,w)
        latent_i = latent_i.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
#         print(latent_i.shape)

        gen_imag = self.decoder(latent_i,out)
        latent_o = self.encoder2(gen_imag)
        
#         print('gen_imag, latent_i, latent_o',gen_imag.shape, latent_i.shape, latent_o.shape)
        return gen_imag, latent_i, latent_o
