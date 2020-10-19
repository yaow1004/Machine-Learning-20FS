# Imports

import argparse
import datetime
import math
import numpy as np
import matplotlib
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

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
		x1 = rand_gaussian(mu[0], sigma[0])
		x2 = rand_gaussian(mu[1], sigma[1])
		data.append([x1, x2])
		labels.append(label)

	return data, labels


def linear_discriminant(x, w):
	"""Linear discriminant function
	"""

	return sum([w[i] * x[i] for i in range(len(x))])


def update_parameter(w, learning_rate, label, x):
	"""Update model parameters
	"""

	return w + learning_rate * label * x


def decision_boundary(x, w):
	"""Decision boundary function
	"""

	return -w[1] / w[2] * x - w[0] / w[2]


def init():

	# 2-d Gaussian parameters: [[mean_0, mean_1], [sigma_0, sigma_1]], with a diagonal covariance matrix

	param_0 = [[15, 10], [12, 12]]
	param_1 = [[-10, -5], [10, 6]]

	# Generate data and labels

	data, labels = generate_data(20, [], [], param_0, 1)
	data, labels = generate_data(30, data, labels, param_1, -1)

	data = np.asarray(data)
	labels = np.asarray(labels)

	# TODO: Compute the required parameters for the quadratic solver

	C = 10  # Regularization parameter for the soft margin
	m,n = data.shape	
	y = labels.reshape(-1,1) *1.  # Data labels
	X_dash = y * data
	H = np.dot(X_dash, X_dash.T)* 1.
	P = cvxopt_matrix(H)
	q = cvxopt_matrix(-np.ones((m,1)))
	G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
	h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m)*C)))
	A = cvxopt_matrix(y.reshape(1,-1))
	b = cvxopt_matrix(np.zeros(1))

	# Optimize

	solution = cvxopt_solvers.qp(P, q, G, h, A, b)

	# Get Lagrangian parameters

	alpha = np.array(solution['x'])
	# TODO: Compute model parameters: w = [intercept, w_1, w_2]

	w =  np.dot((y * alpha).T , data)
	w =  w.flatten()
	b = np.mean(y - np.dot(w, data.T))
	w = [b, w[0], w[1]]

	# Get decision boundary (x, y) pairs

	x = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution for i in range(args.plot_resolution)]
	y = [decision_boundary(xi, w) for xi in x]

	# Plot the decision boundary

	ax = plt.gca()
	ax.plot(x, y, label = 'Decision boundary function')

	# Plot data

	plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], s = 12, c = labels)

	# Draw

	plt.axis(args.plot_boundaries, fontsize = args.font_size)
	plt.title(args.title)
	plt.draw()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex9: Support Vector Machine (Soft-Margin)',
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [-50, 50, -50, 50],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	# LCG

	lcg = LCG(1664525, 1013904223, 2 ** 32)

	init()

