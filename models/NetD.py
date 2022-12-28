import torch
import torch.nn as nn
import math

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
        
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,opt):
        super(NetD, self).__init__()
        
        self.conv1=conv_3x3_bn(inp=3, oup=32, stride=2)
        self.Fused_MBConv_0=MBConv(inp=32, oup=32, stride=1, expand_ratio=1, use_se=0)
        self.Fused_MBConv_1=MBConv(inp=32, oup=64, stride=2, expand_ratio=4, use_se=0)
        self.Fused_MBConv_2=MBConv(inp=64, oup=128, stride=2, expand_ratio=4, use_se=0)
        
        self.MBConv_3=MBConv(inp=128, oup=256, stride=2, expand_ratio=4, use_se=1)
        self.MBConv_4=MBConv(inp=256, oup=512, stride=2, expand_ratio=6, use_se=1)
        self.MBConv_5=MBConv(inp=512, oup=1024, stride=2, expand_ratio=6, use_se=1)
        self.final=nn.Sequential(nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
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
        x= self.MBConv_4(x)
        features = self.MBConv_5(x)#1, 512, 8, 8
#         print(features.shape)
        classifier = self.final(features)#1, 1024, 4, 4
#         print(classifier.shape)
        classifier = classifier.view(-1, 1).squeeze(1)#[1, 1, 1, 1]

        return classifier, features
