#/usr/bin/env python3
'''
created on 6/7 2021
@author=libozhang
'''

import torch
import torch.nn as nn
from typing import Type,List,Union
from torchsummary import summary


def main(model,epochs,num_classes):
    if torch.cuda.is_available():
        device = 'cuda'
    model = model(num_classes)
    #定义优化器和损失函数
    optimizer= torch.optim.Adam(model.parameters(),lr = 0.01,weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()





