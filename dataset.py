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
    transforms.Resize((224,224)),
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def dataset_path(root):
  train_img_path = os.path.join(root,'train')
  test_img_path = os.path.join(root,'test')
  return train_img_path,test_img_path
root = './datasets/horse-human'

train_img_path,test_img_path = dataset_path(root)
train_dataset = ImageFolder(
    root = train_img_path,
    transform=train_transformations,
)
test_dataset = ImageFolder(
    root = test_img_path,
    transform=test_transformations
)

print(f'train_dataset classes:{train_dataset.classes}')
print(f'train_dataset img numbers:{len(train_dataset)}')
print(f'show the dataset {train_dataset.class_to_idx}\n and {train_dataset.imgs[:2]}')

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

if __name__ =="__main__":
    for i,(images,labels) in enumerate(train_loader):
        print(f'input img tensor is {images.size(0)} \n and labels is {labels}')
