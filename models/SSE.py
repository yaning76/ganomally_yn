import torch.nn as nn
import torch
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

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
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ConvSig(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid() if act else nn.Identity()
#         print(autopad(k, p))
    def forward(self, x):
        
        return self.act(self.conv(x))

    
    
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
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#         print(p)
    return p
class Block_Se(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1, padding=1, dilation=1, groups=1,avg_pool=True,
                 se_block=True, activation=nn.SiLU()):

        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
#         self.padding_mode = padding_mode
        self.se_block = se_block
        assert padding == 1
#         padding_11 = padding - 3 // 2

        self.dense_groups = groups
        self.nonlinearity = activation

#         self.rbr_identity = nn.BatchNorm2d(in_channels)
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=self.dense_groups) if (kernel_size != 1) else None

        if self.se_block:
            self.se = SE1(
                in_channels, out_channels, g=groups,
                ver=2 if (out_channels != in_channels or stride != 1) else 1)

    def forward(self, inputs):
        drop_path_output = self.rbr_dense(inputs)
        out = drop_path_output * self.se(inputs)
        out = self.nonlinearity(out)
        return out