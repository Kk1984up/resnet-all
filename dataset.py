# -*-coding: utf-8 -*-
'''
@time :2021/06/09
@author :libo
@file   :dataset
@detail
'''
import os
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#train dataset transformation
train_transformations = transforms.Compose([
    transforms.Tensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    transforms.Resize((224,224)),
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip()
])

test_transformations = transforms.Compose([
    transforms.Tensor(),
    transforms.Resize((224,224)),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def dataset_path(root):
  train_img_path = os.path.join(root,'train')
  test_img_path = os.path.join(root,'test')
  return train_img_path,test_img_path
root = 'd:\\github\\resnet_all\\datasets'

train_img_path,test_img_path = dataset_path(root)
train_img = ImageFolder(
    root = train_img_path,
    transform=train_transformations,
)
test_img = ImageFolder(
    root = test_img_path,
    transform=test_transformations
)

print(f'train_dataset classes:{train_img.classes}')
print(f'train_dataset img numbers:{len(train_img)}')
print(f'show the dataset {train_img.class_to_idx}\n and {train_img.imgs[:2]}')

train_loader = DataLoader(train_img,batch_size=4,shuffle=True)
test_loader = DataLoader(test_img,batch_size=4,shuffle=False)
