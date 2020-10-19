# Imports

import argparse
import datetime
import time
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class LCG:
	"""Linear congruential generator
	"""

	def __init__(self, a, c, m):
		self.rand_number = int(round(m * datetime.datetime.now().second / 60))
		self.a = a
		self.c = c
		self.m = m

	def draw_rand_number(self):
		self.rand_number = (self.a * self.rand_number + self.c) % self.m
		return self.rand_number / self.m


def rand_uniform(a, b):
	"""Draw a random number from an uniform distribution U(a, b)
	"""

	return a + lcg.draw_rand_number() * (b - a)


def rand_gaussian(mu, sigma, n = 0):
	"""Draw random sample from the Gaussian distribution with Box–Muller transform
	"""

	def draw_sample():
		u = rand_uniform(0, 1)
		v = rand_uniform(0, 1)

		return mu + sigma * math.sqrt(-2 * math.log(u)) * math.cos(math.pi * v)

	if n > 0:
		return [draw_sample() for _ in range(n)]

	return draw_sample()


def generate_data(n, data, labels, param, label):
	"""Generate labeled data from a gaussian distributions
	"""

	mu = param[0]
	sigma = param[1]

	for i in range(n):

		# TODO: Recall the notation of x in the exercise sheet.
		# 		Generate a 2-d Gaussian distributed data point plus an offset value for the bias.
		# 		Use our rand_gaussian method.

		x1 = rand_gaussian(mu[0], sigma[0])  
		x2 = rand_gaussian(mu[1], sigma[1])   

		data_point = [1, x1, x2]
		data.append(data_point)
		labels.append(label)

	return data,labels


def linear_discriminant(x, w):
	"""Linear discriminant function
	"""

	# TODO: Implement the linear discriminant function f(x;w) defined in the exercise sheet. 
	w = np.array(w)
	if w.T @ x > 0:   
		return 1
	if w.T @ x < 0:
		return -1


def update_parameter(w, learning_rate, label, x):
	"""Update model parameters
	"""

	# TODO: Implement the parameter update rule. Return the updated parameter.

	return w+x*learning_rate*label


def decision_boundary(x, w):
	"""Decision boundary function
	"""
	# TODO: Derive and implement the decision boundary function for given model parameters w and any x-axis value x
	# np.dot(w.T, x)=0
    # (w[0], w[1],w[2]) * (1,x,y) = 0
    # w[0]*1 +w[1]*x +w[2]*y = 0
    # y = -w[0]/w[2] -w[1]/w[2]*x
	return -w[0]/w[2]-w[1]/w[2]*x

def init():

	# 2d distribution parameters: [[mean_0, mean_1], [sigma_0, sigma_1]],
	# with a diagonal covariance matrix.

	param_0 = [[30, 40], [10, 15]]
	param_1 = [[-10, -5], [10, 6]]

	# Generate input data and labels of size n

	data, labels = generate_data(30, [], [], param_0, 1)
	data, labels = generate_data(20, data, labels, param_1, -1)

	# Initial model parameters

	w = [100, -10, 20]

	# Learning rate

	learning_rate = 1.0

	# Init plot

	ax = plt.gca()

	# Perceptron training loop

	converged = False

	while not converged:

		# Clear previous plot

		plt.cla()

		# Iterate the data

		converged = True

		for x, label in zip(data, labels):
			#  Implement the misclassification error J(w). Use the linear_discriminant method to compute f(x; w).

			if linear_discriminant(x, w) * label < 0:
				error = 1
			else:
				error = 0

			if error > 0:
				# Update all model parameters w

				w = [update_parameter(w[i], learning_rate, label, x[i]) for i in range(len(x))]
				###error += int(np(w.T,x) != 0.0)

				converged = False

		# Plot all observations

		plt.scatter(np.array(data)[:, 1], np.array(data)[:, 2], s = 12, c = labels)

		# Plot decision boundary function

		x = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution for i in range(args.plot_resolution)]
		y = [decision_boundary(xi, w) for xi in x]
		ax.plot(x, y, label = 'Decision boundary function')

		# Redraw

		plt.axis(args.plot_boundaries, fontsize = args.font_size)
		plt.title(args.title)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.025)

	print('Training done')

	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex3: Perceptron Convergence Theorem',
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	### The default settings are cutting off the real data.
	""" Original
	parser.add_argument('--plot-boundaries',
						default = [-10, 10, -10, 10],  # min_x, max_x, min_y, max_y
						required = False)
						"""
	parser.add_argument('--plot-boundaries',
					default = [-20, 50, -20, 70],  # min_x, max_x, min_y, max_y
					required=False)


	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	# LCG

	lcg = LCG(1664525, 1013904223, 2 ** 32)

	init()

