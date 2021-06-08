import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Union
from torchvision.models import resnet152

def conv3x3(ic: int, oc: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    :param ic:input channels
    :param oc: output channels
    :param stride: conv stride
    :param groups: how groups
    :param dilation: conv dilation
    :return: nn.Conv2d
    """
    return nn.Conv2d(ic, oc, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups,bias=False, dilation=dilation)

def conv1x1(ic: int, oc: int, stride: int = 1) -> nn.Conv2d:
    '''
    :param ic: input channels
    :param oc: output channels
    :param stride: move stride
    :return:
    '''
    return nn.Conv2d(ic, oc, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self,
                 inplanes: int,
                 planes: int,






                 ):













