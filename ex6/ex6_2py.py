# Imports

import argparse
import random
import sys

import numpy as np
import matplotlib
import torch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init():

	# Variables

	n = args.n
	res = args.plot_resolution
	lr = args.learning_rate
	epochs = args.epochs

	# Generate Data

	x, y = data_generator(n, second_order_function)

	# Train model parameters

	w = train_nn(x, y, lr, epochs)

	# Predict

	px = [min(x) + i * (max(x) - min(x)) / res for i in range(res)]
	pxx = [[1, elem] for elem in px]
	inputs = torch.from_numpy(np.array(pxx, dtype = 'float32'))

	preds = model(inputs, w)

	print(f'Preds: {preds}')
	print(w)

	py = [elem.item() for elem in preds]

	# Plot

	plot([x, y], [px, py])


def train_nn(x, y, lr, epochs):
	"""Train neural network model
	"""

	# Add x_0 = 1 for the bias

	xx = [[1, elem] for elem in x]
	targets = [[elem] for elem in y]

	# Torch tensor

	inputs = torch.from_numpy(np.array(xx, dtype = 'float32'))
	targets = torch.from_numpy(np.array(targets, dtype = 'float32'))

	# Define model

	# TODO: Initialize random model perameters for each neuron.
	# Use the torch package to initialize a tensor with a required gradient flag
	# Note, the input layer is just the data X and has no trainable parameters.
#	dtype = torch.float
#	device = torch.device("cpu")
#   N, D_in, H, D_out = 200,2,2,1...
    #x = torch.randn(N,D_in, dtype=dtype, device=device)
    #H_00 = torch.randn(D_in, H,dtype=dtype, device=device)...
	H_00 = torch.randn(2, requires_grad=True)
	H_01 = torch.randn(2, requires_grad=True)
	H_10 = torch.randn(2, requires_grad=True)
	H_11 = torch.randn(2, requires_grad=True)
	O_0 = torch.randn(2, requires_grad=True)

	# A neural network consisting of 3 layers.

	network = [[H_00, H_01], [H_10, H_11], [O_0]]

	# Train model parameters
	for i in range(epochs):

		print(f'network: {network}')

		# TODO: Compute predictions y. Use the model(x, network) method.

		y = model(inputs, network)

		#print(y)

		# TODO: Compute the squared error between targets and y outputs. Use the squared_error method.

		loss = squared_error(targets, y)

		# Compute the gradient for each trainable parameter in the network.

		loss.backward()

		print('SE: {}'.format(loss.item()))

		with torch.no_grad():
			for layer in network:
				for neuron in layer:
					# TODO: Update parameters of each neuron. Access the gradient with: neuron.grad, involve the learning rate lr.
					neuron -= lr*neuron.grad
					# Reset the gradients
					neuron.grad.zero_()

	return network


def model(x, network):
	"""Multilayer feed-forward neural network
	"""

	# Network layers

	H0 = network[0]
	H1 = network[1]
	O = network[2]

	# TODO: Feed forward the signal x in order to get an output y.
	# Use hyperbolic tangent activation function. In the output player, use a linear activation.
	# Concatenate neuron outputs after each layer for a concise notation.

	H_00 = torch.tanh(H0[0][0]*x[:,0]+H0[0][1]*x[:,1])
	H_01 = torch.tanh(H0[1][0]*x[:,0]+H0[1][1]*x[:,1])
	H_0_out = H_00 + H_01

	H_10 = torch.tanh(H1[0][0]*H_0_out)
	H_11 = torch.tanh(H1[1][1]*H_0_out)
	H_1_out = H_10 + H_11

	y_hat = (O[0][0]+O[0][1])*H_1_out

	return y_hat



def squared_error(y, y_hat):

	# TODO: Compute the mean squared error

	return torch.sum((y - y_hat).pow(2)/2)


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
						default = 'Ex6.2: Neural Network',
						required = False)

	parser.add_argument('--n',
						default = 200,
						required = False)

	parser.add_argument('--learning-rate',
						default = 0.0005,
						required = False)

	parser.add_argument('--epochs',
						default = 1000,
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