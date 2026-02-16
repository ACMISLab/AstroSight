#%%

        
import torch
import torch.nn.functional as F

import torchvision.ops
from torch import nn
import math
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

# 设置随机数种子
setup_seed(20) 
class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):

        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=True)
        
        # init        
        self.reset_parameters()
        self._init_weight()


    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        self.bias.data.zero_()


    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.weight, 
                                          bias=self.bias, 
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x


class ResBlock_dcn(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1,dcn=True):
        super(ResBlock_dcn, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.dcn=dcn

        if self.dcn==True:
            self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            DCNv2(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        else:
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )





        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_channels):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        proj_query_normalized = torch.nn.functional.normalize(proj_query, dim=-1)
        proj_key_normalized = torch.nn.functional.normalize(proj_key, dim=-2)
        energy = torch.bmm(proj_query_normalized, proj_key_normalized)

        # energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)

class resnet_HAN_DCN(nn.Module):
    def __init__(self, num_classes=7):
        super(resnet_HAN_DCN, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )


        self.layer2 = self.make_layer(ResBlock_dcn, 128, 2, stride=2,dcn=True)
        self.layer3 = self.make_layer(ResBlock_dcn, 256, 2, stride=2,dcn=True)       
        self.la3 = LAM_Module(256)
        self.lastconv3=nn.Conv2d(256*2,256,3,1,1)


        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),

        )
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride,dcn=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride,dcn=dcn))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward_2(self, x):  
        out1 = self.conv1(x)


        out1 = self.layer2(out1)

        return out1    
    def forward_3(self, x):  
        for name, midlayer in self.layer3._modules.items():

            if name=='0':
                res = midlayer(x)
                res1 = res.unsqueeze(1)
                out_1=res
                # print(res1.shape)
            else:
                res = midlayer(res)

                res1 = torch.cat([res.unsqueeze(1),res1],1)
        res=self.la3(res1)
        out2=self.lastconv3(res)

        return out2  
    
  
    def forward(self, x):  
        
        out=self.forward_2(x)
        out=self.forward_3(out)

        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

