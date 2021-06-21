#/usr/bin/env python3
'''
create on 6/16 2021
@author=libo
'''
from torch._C import device
import torch.nn as nn
import torch
from  dataset import train_loader,test_loader
from model.resnet_all import resnet101,resnet152



 
def train_one_epoch(dataloader,model,loss_fn,optimizer):
	size = len(dataloader.dataset)
	print(f'dataloader size is {size}')
	for batch,(image,label) in enumerate(dataloader):
		print(f'batch is {batch}')
		# print(f'batch is :{batch} and \n image:{image} \n label:{label}')
		image,label = image.to(device),label.to(device)
		# compute prediction error
		pred = model(image)
		loss = loss_fn(pred,label)
		#backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch %16 ==0:
			loss,current = loss.item(), batch*len(image)
			print(f'loss:{loss}  [{current}/{size}]')


def test(dataloader,model):
	size = len(dataloader.dataset)
	model.eval()
	best_acc =0
	test_loss,correct = 0,0
	with torch.no_grad():
		for image,label in dataloader:
			image,label = image.to(device),label.to(device)
			print(f'batch is :{image.size(0)} and \n image:{image}')
			pred = model(image)

			print(f'pred is :{pred} \n and label is {label}')
			test_loss += loss_fn(pred,label).item()
			print(f'')
			correct+= (pred.argmax(1) ==label).type(torch.float).sum().item()
		test_loss/=size
		correct/= size
		print(f'Test Error:\n Accuracy:{(100*correct):>0.1f}%, Avg loss:{test_loss:>8f}\n')
	return correct


epochs =100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')
print(f'device count num :{torch.cuda.current_device()}')
torch.cuda.set_device(1)
print(f'current device is :{torch.cuda.current_device()}')
model = resnet152(num_classes =2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.0001)

def main():
	for epoch in range(epochs):
		best_acc = 0
		print(f'Epoch {epoch+1}\n -------------------')
		train_one_epoch(train_loader,model,loss_fn,optimizer)
		correct = test(test_loader,model)
		if correct>best_acc:
			torch.save(model.state_dict(),'model_best.pth')
	print('Done')

if __name__ =='__main__':
	main()




		






