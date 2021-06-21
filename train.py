#/usr/bin/env python3
'''
create on 6/18 2021
@author=libo
'''

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import torchvision
import time
import copy
from model.resnet_all import resnet152
from torch.optim import SGD,lr_scheduler

args = argparse.ArgumentParser(description = 'train your pytorch')
args.add_argument('-i','--input',type = str,default = 'datasets',help = 'path to input images path')
args.add_argument('-o','--output',type = str,default='./results',help = 'save path to output model file ')
args.add_argument('-e','--epochs',type = int, default=80,help = 'numbers of trainning max epochs')
args.add_argument('-n','--num_classes',type= int,default = 10,help = 'numbers of classes about dataset')
args.add_argument('-l','--learning_rate',type = float,default=0.001,help = 'the learning rate in training')
args.add_argument('-b','--batch_size',type = int, default=16,help = 'batch size for dataloader')

if torch.cuda.is_available():
	gpus = torch.cuda.device_count()
	print(f'numbers of  gpu is {gpus}')

ar = args.parse_args()
data_dir = ar.input
def path2data(sub):
	return Path(data_dir)/sub


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def dataloader(data_dir,batch_size=ar.batch_size,train=True):
	if train:
		data_transform= transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
		])
	else:
		data_transform=transforms.Compose([
			transforms.Resize(248),
			transforms.CenterCrop(),
			transforms.ToTensor(),
			transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
		])
	images_datasets = ImageFolder(data_dir,data_transform)
	dataloader = DataLoader(images_datasets,batch_size=batch_size,shuffle=train,num_workers=16)
	classes_name = images_datasets.classes
	dataset_size = len(images_datasets)
	print(f'dataset_size is :{dataset_size}\n and classes_name:{classes_name}')
	return dataset_size,classes_name,dataloader

def imshow(inp,title =None):
	inp = inp.numpy().transpose((1,2,0))
	mean = np.array([0.485,0.456,0.406])
	std = np.array([0.229,0.224,0.225])
	inp = std*inp +mean
	inp = np.clip(inp,0,1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.01)

_,class_names, val_loader = dataloader(path2data('val'),ar.batch_size)
inp,classes = next(iter(val_loader))
out = torchvision.utils.make_grid(inp)
imshow(out,title = [class_names[x] for x in classes])

def getloaders():
	dataloders = {x:dataloader(path2data(x))[2] for x in ['train','val']}
	print(f'the dataloaders show :{dataloders}')
	return dataloders
dataloaders = getloaders()

def dataset_size():
	size = {x:dataloader(path2data(x))[0] for x in ['train','val']}
	print(f'dataset size is :{size}')
	return size

dataset_sizes = dataset_size()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(f'outputs is :{outputs} \n and preds is :{preds}')
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                print(f'preds is {preds}\n and labels is :{labels.data}')
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                visualize_model(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():	
    model_ft = resnet152(num_classes =4)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
	# Observe that all parameters are being optimized
    optimizer_ft = SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
	# Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=ar.epochs)
    visualize_model(model_ft,num_images=12)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
	main()