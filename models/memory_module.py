from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
#         print(input.shape)
#         print(self.weight.shape)
        
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
#             att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        x = input.permute(0, 2, 3, 1)
#         print(x.shape)
            
        x = x.contiguous()
#         print(x.shape)
        x = x.view(-1, s[1])
#         print(x.shape)
        #
        y_and = self.memory(x)
        
        #
        y = y_and['output']
#         print('y',y.shape)
        att = y_and['att']
#         print('att',att.shape)
        
        y = y.view(s[0], s[2], s[3], s[1])
#         print( y.shape)
        y = y.permute(0, 3, 1, 2)
#         print( y.shape)
        att = att.view(s[0], s[2], s[3], self.mem_dim)
#         print( att.shape)
        att = att.permute(0, 3, 1, 2)
#         print( att.shape)

        return {'output': y, 'att': att}

####################################################
def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val

class MemModule1(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule1, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, x):
        s = x.data.shape
#         x = input.permute(0, 2, 3, 1).contiguous()
#         print(x.shape)
        x = x.view(s[1]*s[0], -1)
#         print(x.shape)
#         print("self.fea_dim.shape",self.fea_dim)
        #
        y_and = self.memory(x)
        
        #
        y = y_and['output']
#         print('y',y.shape)
        att = y_and['att']
#         print('att',att.shape)
        
        y = y.view(s[0], s[1], s[2], s[3])
#         print( y.shape)
#         y = y.permute(0, 3, 1, 2)
#         print( y.shape)
        att = att.view(s[0],s[1],self.mem_dim)
#         print( att.shape)
#         att = att.permute(0, 3, 1, 2)
#         print( att.shape)

        return {'output': y, 'att': att}

# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
class MemModule_w(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025,window_size=2, device='cuda'):
        super(MemModule_w, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        self.window_size=window_size
    def forward(self, input):
        s = input.data.shape
#         print(s)
        x = input.permute(0, 2, 3, 1).contiguous()
#         print(x.shape)
        x=window_partition(x, self.window_size)###########
#         print(x.shape)
        b,w,w,c=x.shape
        x = x.view(b, -1)
#         x = x.view(-1, s[1])
#         print(x.shape)
        #
        y_and = self.memory(x)
        y = y_and['output']
#         print('y',y.shape)
        att = y_and['att']
#         print('att',att.shape)
        y=window_reverse(y,self.window_size,s[2], s[3])########
#         print( y.shape)
        y = y.permute(0, 3, 1, 2)
#         print( y.shape)
        att=window_reverse(att,self.window_size,s[2], s[3])
        att = att.permute(0, 3, 1, 2)
#         print( att.shape)

        return {'output': y, 'att': att}


class MemModule_w_new(nn.Module):
    def __init__(self, mem_dim, fea_dim, window,c_size=1,shrink_thres=0.0025, device='cuda'):
        super(MemModule_w_new, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        self.window=window
        self.c_size=c_size
    def forward(self, input1):
        if self.c_size==1:
            s = input1.data.shape
            x = input1.permute(0, 2, 3, 1).contiguous()
            b,h,w,c=x.shape
            num_window=int((h/self.window)*(w/self.window))
            x=x.view(b*num_window,self.window,self.window,c)
            x=x.view(x.size(0), -1)

            y_and = self.memory(x)

            y = y_and['output']
            att = y_and['att']
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], int(h/self.window), int(w/self.window), self.mem_dim)
            att = att.permute(0, 3, 1, 2)
    #         print("att.shape",att.shape)
            return {'output': y, 'att': att}
        else:
            B,C,H,W=input1.shape
            input1=input1.view(B*self.c_size,-1,H,W)
            s = input1.data.shape
            x = input1.permute(0, 2, 3, 1).contiguous()
            b,h,w,c=x.shape
            num_window=int((h/self.window)*(w/self.window))
            x=x.view(b*num_window,self.window,self.window,c)
            x=x.view(x.size(0), -1)
            y_and = self.memory(x)
            y = y_and['output']
#             print(y.shape)
            att = y_and['att']
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2).contiguous()
            y = y.view(B,C,H,W)
            
#             print(att.shape)
            
            att = att.view(B,self.c_size*int(h/self.window)*int(w/self.window), self.mem_dim)
#             att = att.permute(0, 3, 1, 2)
            return {'output': y, 'att': att}

class MemModule1_new(nn.Module):
    def __init__(self,mem_dim,fea_dim,window=1,shrink_thres=0.0025, device='cuda'):
        super(MemModule1_new, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        self.window=window

    def forward(self, x):
        s = x.data.shape
        window_size=s[2]//self.window
        if self.window!=1:
            x = x.view(s[0],s[1],self.window,window_size,self.window,window_size)
            x=x.permute(0, 1, 2, 4,3,5).contiguous().view(-1, window_size*window_size)
            y_and = self.memory(x)
            y = y_and['output']
            att = y_and['att']
            y = y.view(s[0],s[1],self.window,self.window,window_size,window_size)
            y=y.permute(0, 1, 2, 4,3,5).contiguous().view(s[0], s[1], s[2], s[3])
            att = att.view(s[0],s[1],self.window,self.window,self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            x = x.view(s[0]*s[1],-1)
            y_and = self.memory(x)
            y = y_and['output']
            att = y_and['att']
            
            y = y.view(s[0],s[1],s[2],s[3])
            att = att.view(s[0],s[1],self.mem_dim)
            att = att.permute(0, 2,1)
        return {'output': y, 'att': att}


def window_partition_c(x, window_size,c_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C//c_size,c_size)
    windows = x.permute(0, 1, 3,5, 2, 4, 6).contiguous().view(-1, window_size, window_size, c_size)
    return windows


def window_reverse_c(windows, window_size,c_size,B, H, W,C):
    x = windows.view(B, H // window_size, W // window_size, C // c_size, window_size, window_size,  c_size)
    x = x.permute(0, 1, 4, 2, 5, 3,6).contiguous().view(B, H, W, C)
    return x
#终版
class MemModule_window(nn.Module):
    def __init__(self, mem_dim, fea_dim,window,c_num, shrink_thres=0.0025, device='cuda'):
        super(MemModule_window, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
        self.window=window
        self.c_num=c_num
    def forward(self, input):
        s = input.data.shape
        x = input.permute(0, 2, 3, 1).contiguous()
        c_size=s[1]//self.c_num
        if self.c_num==1:
            x=window_partition(x,self.window)
            num_window=int((s[2]/self.window)*(s[3]/self.window))
            x=x.view(s[0]*num_window,-1)

            y_and = self.memory(x)
            y = y_and['output']
            att = y_and['att']
            y = window_reverse(y,self.window,s[2],s[3])
            y = y.permute(0, 3, 1, 2)
            att=att.view(s[0], s[2]//self.window, s[3]//self.window, self.mem_dim).permute(0, 3, 1, 2)
            return {'output': y, 'att': att}
        else:
            x=window_partition_c(x,self.window,c_size)
            num_window=(s[2]//self.window)*(s[3]//self.window)*self.c_num
            x=x.view(s[0]*num_window,-1)

            y_and = self.memory(x)
            y = y_and['output']
            att = y_and['att']
            y = window_reverse_c(y,self.window,c_size,s[0],s[2],s[3],s[1])#windows, window_size,c_size,B, H, W,C
            y = y.permute(0, 3, 1, 2)
            att=att.view(s[0], s[2]//self.window, s[3]//self.window,self.c_num, self.mem_dim).permute(0,4,1,2,3)
            return {'output': y, 'att': att}


#最终版 有错误
# class MemModule1_new(nn.Module):
#     def __init__(self,mem_dim,fea_dim,window=1,shrink_thres=0.0025, device='cuda'):
#         super(MemModule1_new, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.shrink_thres = shrink_thres
#         self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
#         self.window=window

#     def forward(self, x):
#         if self.window!=1:
#             s = x.data.shape
#             x = x.view(s[0],s[1],s[2]//self.window,self.window,s[3]//self.window,self.window)
#             x=x.permute(0, 1, 3, 5,2,4).contiguous().view(-1, s[2]//self.window*s[3]//self.window)
#             y_and = self.memory(x)
#             y = y_and['output']
#             att = y_and['att']
    
#             y = y.view(s[0],s[1],self.window,self.window,s[2]//self.window,s[3]//self.window)
#             y=y.permute(0, 1, 4, 2,5,3).contiguous().view(s[0], s[1], s[2], s[3])
            
#             att = att.view(s[0]*s[1],self.window,self.window,self.mem_dim)
#             att = att.permute(0, 3, 1, 2)
#         else:
#             s = x.data.shape
#             x = x.view(s[0]*s[1],-1)
#             y_and = self.memory(x)
#             y = y_and['output']
#             att = y_and['att']
    
#             y = y.view(s[0],s[1],s[2],s[3])
            
#             att = att.view(s[0]*s[1],self.window,self.window,self.mem_dim)
#             att = att.permute(0, 3, 1, 2)
            
#     #         att = att.view(s[0]* s[1],self.window, self.window, self.mem_dim)
#         return {'output': y, 'att': att}




#终版,有错误
# class MemModule_window(nn.Module):
#     def __init__(self, mem_dim, fea_dim,window,c_size=1, shrink_thres=0.0025, device='cuda'):
#         super(MemModule_window, self).__init__()
#         self.mem_dim = mem_dim
#         self.fea_dim = fea_dim
#         self.shrink_thres = shrink_thres
#         self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)
#         self.window=window
#         self.c_size=c_size
#     def forward(self, input):
#         if self.c_size==1:
#             s = input.data.shape
#             x = input.permute(0, 2, 3, 1).contiguous()
#             x=window_partition(x,self.window)
#             num_window=int((s[2]/self.window)*(s[3]/self.window))
#             x=x.view(s[0]*num_window,-1)

#             y_and = self.memory(x)
#             y = y_and['output']
#             att = y_and['att']

#             y = window_reverse(y,self.window,s[2],s[3])
#             y = y.permute(0, 3, 1, 2)
#             att = att.view(s[0], s[2] // self.window, s[2] // self.window,self.mem_dim)
#             att = att.permute(0, 3, 1,2)

#             return {'output': y, 'att': att}
#         else:
#             b,c,h,w=input.shape
#             x=input.view(b,c//self.c_size,self.c_size,h,w)
#             x=x.permute(0, 2,1,3,4).contiguous()
#             x=x.view(b*self.c_size,c//self.c_size,h,w)
            
#             s = x.data.shape
#             x = x.permute(0, 2, 3, 1).contiguous()
#             x=window_partition(x,self.window)
#             num_window=int((s[2]/self.window)*(s[3]/self.window))
#             x=x.view(s[0]*num_window,-1)

#             y_and = self.memory(x)
#             y = y_and['output']
#             att = y_and['att']
#             y = window_reverse(y,self.window,s[2],s[3])
#             y = y.permute(0, 3, 1, 2).contiguous()
#             y=y.view(b,self.c_size,c//self.c_size,h,w)
#             y=y.permute(0, 2,1,3,4).contiguous()
#             y=y.view(b,c,h,w)
            
#             att = att.view(s[0], s[2] // self.window, s[2] // self.window,self.mem_dim)
#             att = att.permute(0, 3, 1,2)
# #             print(att.shape)
#             return {'output': y, 'att': att}
