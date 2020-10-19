# Imports

import argparse
import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init():

	# Variables

	n = args.n
	d = args.polynomial_order
	res = args.plot_resolution

	# Generate Data

	x, y = data_generator(n, third_order_function)

	# Polynomial regression

	beta, sigma = polynomial_regression(x, y, d)

	# Evaluate regression function

	px = [min(x) + i * (max(x) - min(x)) / res for i in range(res)]
	py = polynomial_value(px, beta)

	# Plot

	plot([x, y], [px, py])


def third_order_function(x):
    # A polynomial basis expansion of a polynomial degree d − 1
   #here is the third order, so d-1 = 3, d = 4
	return x ** 3 - x




def polynomial_value(x, beta):
	"""Evaluate polynomial at data points X, given parameters beta
	"""

	# TODO: Evaluate a polynomial function for data points x


	d = len(beta)
	X = np.vander(x, d, increasing=True)
	return beta @ X.T		


def data_generator(n, f):
	"""Generate training data
	"""

	x = [3 * (random.random() - 0.5) for _ in range(n)]
	y = [f(x[i]) + 2 * (random.random() - 0.5) for i in range(len(x))]

	return x, y

def polynomial_regression(x, y, d):
	"""Polynomial regression
	"""
	# TODO: Compute the expansion of x in the polynomial basis
	n = len(x)
	x = np.array(x)   # create an array for x vectors
	y = np.array(y)   # create an array for y vectors

	X = np.vander(x, d, increasing=True)   # generate a Vandermonde matrix for X
      # TODO: Compute the estimators for beta

	beta = np.linalg.inv(X.T @ X) @ X.T @ y #np.linalg.inv() -Compute the (multiplicative) inverse of a matrix.

	# TODO: Compute the estimator for sigma
	sigma = np.sqrt(1/n * (y - X @ beta).T @ (y - X @ beta))#np.sqrt()-return the positive square-root of an array

	return beta, sigma


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
						default = 'Ex4: Polynomial Regression',
						required = False)

	parser.add_argument('--n',
						default = 50,
						required = False)

	parser.add_argument('--polynomial-order',
						default = 4,
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