#/usr/bin/env python3
'''
created on 6/7 2021
@author=libozhang
'''
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter
from dataset import train_loader,test_loader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.resnet_all import resnet101,resnet152
from torch.autograd import Variable




model = resnet101(num_classes =2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=model.to(device)
#定义优化器和损失函数
optimizer= Adam(model.parameters(),lr = 0.01,weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(num_epochs):
    best_acc =0
    for epoch in range(num_epochs):
        model.train()
        train_acc =0
        train_loss =0
        for i,(images,labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images)
                labels = Variable(labels)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()

        optimizer.step()
        if epoch%10==0:
            torch.save(model.state_dict(),f'resnet_{epoch}epoch.pth')
        print(f'{epoch}/{num_epochs}: loss is :{loss}')


def main():
    train(100)

if __name__=="__main__":
    main()








