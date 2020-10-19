# Imports

import argparse
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as torch_data
import torchvision.transforms as transforms
from torch.autograd import  Variable
import sys
from collections import defaultdict
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class NetSeq(nn.Module):
	def __init__(self):
		super(NetSeq, self).__init__()

		self.cnn_layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1), nn.ReLU(),nn.MaxPool2d(2,2))
		self.cnn_layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),nn.MaxPool2d(2,2))
		self.drop1 = nn.Dropout(p=0.2)
        
		# TODO: Initialize fully connected layer section using a sequential container.

		self.fc_layer1 = nn.Sequential(nn.Linear(5*5*64,128),nn.BatchNorm1d(128), nn.ReLU())
		self.drop2 = nn.Dropout(p=0.2)
		self.fc_layer2 = nn.Linear(128,10)

			# TODO: What is the input dimension of the first fully connected layer? Calculate the result manually.
	def forward(self, x):
		# TODO: Forward the input x.
		x = self.cnn_layer1(x)
		x = self.cnn_layer2(x)
		x = x.view(x.size(0),-1)
		x = self.fc_layer1(x)
		x = self.fc_layer2(x)
		return F.log_softmax(x, dim=1)


def init():

	# Variables

	classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
	lr = args.learning_rate
	epochs = args.epochs
	batch_size = args.batch_size

	# Load data

	transform = transforms.Compose([transforms.ToTensor()])

	trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
	trainloader = torch_data.DataLoader(trainset, batch_size = batch_size, shuffle = True)

	testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

	# Train model or load a trained model

	if not args.model_path:
		network = train_nn(trainloader, testloader, classes, batch_size, lr, epochs)
	else:
		network = NetSeq()
		# TODO: Load stored model weights.
		try:
			network = NetSeq()
			network.load_state_dict(torch.load(args.model_path))
			network.eval()
		except FileNotFoundError as e:
			print(e)
			sys.exit()
		print(f' Network {args.model_path} was successfully loaded!')
	# Classify

	classify(network, testloader, testset, classes)


def train_nn(trainloader, testloader, classes, batch_size, lr, epochs):
	"""Train neural network model
	"""

	network = NetSeq()

	# TODO: Initialize cross entropy loss function
	criterion = nn.CrossEntropyLoss()

	# TODO: Initialize Adam optimizer
	optimizer = torch.optim.Adam(network.parameters(), lr=lr)

	print('Start training')

	for epoch in range(epochs):
		train_loss = 0
		acc_total = 0
		print('Epoch: {}'.format(epoch + 1))
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = network(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
		# Validate all classes
		# TODO: Validate the model on the validation data. Print out the overall network accuracy.
		# TODO: Validate the model on the validation data. Print out the network accuracy for each class separately.
		# Validate each class

			train_loss += loss.item()+inputs.size(0) #compute loss
			if i %999 == 0: # print out the average loss every 1000 steps
			    print('epoch:%d, batch:%d, loss:%.3f' % (epoch+1, i+1,train_loss / 1000))
			_,pred = torch.max(outputs.data,1)
			correct = (pred == labels).sum().item()
			acc_total = round(correct / (len(trainloader) * batch_size) , 4)
			print(f'Correct Test: {correct} Accuracy: {acc_total}')
		# Save the model
		torch.save(network.state_dict(), './mnist_epoch_{}_acc_{}.pth'.format(epoch + 1, acc_total))

	return network


def classify(network, testloader, testset, classes):
	"""Classify input
	"""
	print("Start classification \n")
	false_classified = defaultdict(list) 
	correct = 0

	for i, data in enumerate(testloader):
		inputs, labels = data  
		outputs = network(inputs)
		prediction = outputs.argmax(1) 

		correct += prediction.eq(labels).sum().item()
		diff = prediction.eq(labels)
		index_diff = np.where(diff == False)[0]
		# Save all missclassified digit index
		for index in index_diff:
			l = labels[index].item()
			false_classified[l].append(index + i * 10)

	acc_total = correct / (len(testloader) * 10)
	print("############## Statistic test set ##############")
	print(f'Total correct: {correct} Accuracy: {acc_total}')

	for i in sorted(false_classified):
		print(f'False digit: {i} Total: {len(false_classified[i])} ({len(false_classified[i]) / (len(testloader))})')

	# Use torchvision package to plot the images
	plot_missclassified(false_classified, testset)


def plot_missclassified(false_classified, image_set):

	for digit_class in sorted(false_classified):
		for i, nr in enumerate(false_classified[digit_class]):
			if i == 10: break

			image, _ = image_set[nr]
			plt.subplot(10, 10, digit_class*10+i+1)
			plt.axis('off')
			plt.imshow(image.numpy().squeeze(), cmap='gray_r')

	plt.show()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'False Classifications for each Class',
						required = False)

	parser.add_argument('--learning-rate',
						default = 0.001,
						required = False)

	parser.add_argument('--epochs',
						default = 4,
						required = False)

	parser.add_argument('--model-path',
						default = '',
						required = False)

	parser.add_argument('--batch-size',
						default = 10,
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	init()
