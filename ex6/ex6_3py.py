# Imports

import argparse
import random
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Net(nn.Module):
	def __init__(self, input_dim, hidden_0_dim, hidden_1_dim, hidden_2_dim, output_dim):
		super(Net, self).__init__()

		# TODO: Implement linear input, hidden, and output layers

		self.input = torch.nn.Linear(input_dim, hidden_0_dim)
		self.hidden_1 = torch.nn.Linear(hidden_0_dim, hidden_1_dim)
		self.hidden_2 = torch.nn.Linear(hidden_1_dim, hidden_2_dim)
		self.output = torch.nn.Linear(hidden_2_dim, output_dim)
		#self.output = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_0_dim), torch.nn.Linear(hidden_0_dim, hidden_1_dim),torch.nn.Linear(hidden_1_dim, hidden_2_dim),torch.nn.Linear(hidden_2_dim, output_dim))
        # refer: https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/303_build_nn_quickly.py (line12-22)
        # self.hidden =torch.nn.Linear (n_feature, n_hidden)
        # self.predict =torch.nn.Linear(n_hidden, n_output)
  
	def forward(self, x):

		# TODO: Forward the signal x. Use ReLu activation. In the output player, use a linear activation.
		x = F.relu(self.input(x))
		x = F.relu(self.hidden_1(x))
		x = F.relu(self.hidden_2(x))
		y_hat = self.output(x)
# refer: https://discuss.pytorch.org/t/implementing-rnn-and-lstm-into-dqn-pytorch-code/14262
        # notice the letters of "relu", Function. not attribute to "ReLU"
		return y_hat


def init():

	# Variables

	n = args.n
	res = args.plot_resolution
	lr = args.learning_rate
	epochs = args.epochs

	# Generate Data

	x, y = data_generator(n, second_order_function)

	# Train model parameters

	model = train_nn(x, y, lr, epochs)

	# Predict

	px = [min(x) + i * (max(x) - min(x)) / res for i in range(res)]
	pxx = [[elem] for elem in px]
	inputs = torch.from_numpy(np.array(pxx, dtype='float32'))

	preds = model(inputs)
	py = [elem.item() for elem in preds]

	# Plot

	plot([x, y], [px, py])


def train_nn(x, y, lr, epochs):
	"""Train neural network model
	"""

	# Pre-process data

	xx = [[elem] for elem in x]
	targets = [[elem] for elem in y]

	inputs = torch.from_numpy(np.array(xx, dtype = 'float32'))
	targets = torch.from_numpy(np.array(targets, dtype = 'float32'))

	# Initialize model

	network = Net(1, 64, 32, 16, 1)

	# TODO: Initialize the mean squared loss. Use a PyTorch method.
    #define the model
	#model = torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(16,1))
	loss_fn = torch.nn.MSELoss(reduction = 'sum')
    #define the loss function
    #nn.MSELoss() is implemented by default as ((input -target)**2).mean()

	# TODO: Initialize the ADAM optimizer. Use a PyTorch method.
	optimizer = torch.optim.Adam(network.parameters(),lr = lr)
    # notice the parameters -> "network.parameters", not the "network.parameters"
    #refer https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html

	# Train model parameters

	for i in range(epochs):
		optimizer.zero_grad()  # Reset the gradients

		# TODO: Compute the network predictions

		outputs = network.forward(inputs)

		# TODO: Compute the prediction loss

		loss = loss_fn(outputs, targets)

		# TODO: Compute the gradients. Use a PyTorch method.

		optimizer.zero_grad()
        #zero weight gradients with w_.grad_zero_()

		# TODO: Update the model parameters. Use a PyTorch method.

		loss.backward()
		optimizer.step()

		print('MSE: {}'.format(loss.item()))

	return network


def second_order_function(x):

	return x ** 2 - x


def data_generator(n, f):
	"""Generate training data
	"""

	x = [3 * (random.random() - 0.5) for _ in range(n)]
	y = [f(x[i]) + 1 * (random.random() - 0.5) for i in range(len(x))]

	return x, y


def plot(observations, poly_estimate):
	"""Plot data and regression
	"""

	plt.rc('xtick', labelsize = args.font_size)
	plt.rc('ytick', labelsize = args.font_size)
	plt.figure(figsize = (4.8, 4))
	plt.title(args.title)
	plt.axis(args.plot_boundaries, fontsize = args.font_size)
	plt.plot(poly_estimate[0], poly_estimate[1], color = 'red')
	plt.scatter(observations[0], observations[1], s = args.scatter_size)
	plt.xlabel('X', fontsize = args.font_size)
	plt.ylabel('Y', fontsize = args.font_size)
	plt.show(block = True)
	plt.interactive(False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex6.3: Neural Network',
						required = False)

	parser.add_argument('--n',
						default = 100,
						required = False)

	parser.add_argument('--learning-rate',
						default = 0.001,
						required = False)

	parser.add_argument('--epochs',
						default = 500,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [-1.5, 1.5, -1.5, 3],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--scatter-size',
						default = 20,
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()
	init()
