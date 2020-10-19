# Imports

import argparse
import time
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init():

	# Variables

	n = args.n
	std_y = 0.35  # Observation noise
	std_f = 5.0  # Prior variance constant

	# Sampled observations

	x_train = np.array([])  # Observed data x
	y_train = np.array([])  # Observed corresponding y
	x = np.arange(-1.5, 1.6, 0.1).reshape(-1, 1)  # Unobserved data points

	# Init plot

	ax = plt.gca()

	# Sampling loop

	for i in range(n):
		# Clear previous plot

		plt.cla()

		# Add a new observation to the sample

		x_train, y_train = add_observation_point(x_train, y_train, std_y, third_order_function)

		# Posterior predictive

		mu, cov = posterior_predictive(x, x_train, y_train, l = 1, std_f = std_f, std_y = std_y)
		mu = mu.ravel()   #mean must be 1 dim
		# Sample from the posterior

		# TODO: Sample 2 functions from the posterior predictive. Use a multivariate normal distribution.

		samples = np.random.multivariate_normal(mu, cov,2)
        
		# Uncertainty boundaries

		uncertainty = 1.96 * np.sqrt(np.diag(cov))

		# Plot

		plt.axis(args.plot_boundaries, fontsize = args.font_size)
		plt.fill_between(x.ravel(), mu + uncertainty, mu - uncertainty, alpha = 0.1, label = 'Standard deviation')
		plt.scatter(x_train, y_train, s = args.scatter_size, label = 'Observations')
		plt.plot(x.ravel(), mu, label = 'Mean estimate')

		# Plot sampled functions

		# TODO: Plot sampled functions sample[0], sample[1]
        # plot 2 random samples
		plt.plot(x, samples[0], label='Sample 0',color='blue', marker='o', linestyle='dashed',linewidth=2, markersize=2)
		plt.plot(x, samples[1], label='Sample 1', color='orange', marker='o', linestyle='dashed',linewidth=2, markersize=2)
		# Plot settings

		ax.legend(loc = 'upper right')
		plt.rc('xtick', labelsize = args.font_size)
		plt.rc('ytick', labelsize = args.font_size)
		plt.title(args.title)
		plt.xlabel('X', fontsize = args.font_size)
		plt.ylabel('Y', fontsize = args.font_size)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.5)

	plt.show()


def kernel(x, x_prime, l, std_f):
	"""Squared exponential kernel
	"""

	# TODO: Implement the squared exponential kernel

	exp = np.sum(x**2, 1).reshape(-1, 1) + np.sum(x_prime**2, 1) - 2 * np.dot(x, x_prime.T)
	return std_f * np.exp(-0.5 / l**2 * exp)


def posterior_predictive(x, x_train, y_train, l, std_f, std_y):
	"""GP posterior predictive distribution
	"""

	# TODO: Implement the posterior predictive parametrization

	K = kernel(x_train, x_train, l, std_f) + std_y**2 * np.eye(len(x_train))
	K_s = kernel(x_train, x, l, std_f)
	K_ss = kernel(x, x, l, std_f) + std_y * np.eye(len(x))
	K_inv = np.linalg.inv(K)
    
	mu = K_s.T.dot(K_inv).dot(y_train)

	cov = K_ss - K_s.T.dot(K_inv).dot(K_s)

	return mu, cov


def add_observation_point(x, y, std_y, f):
	"""Add a new observation
	"""

	xp = np.random.uniform(-1.5, 1.5, 1)
	yp = f(xp[0]) + np.random.normal(0, std_y, 1)
	x = np.append(x.ravel(), xp)
	y = np.append(y.ravel(), yp)

	return x.reshape(-1, 1), y.reshape(-1, 1)


def third_order_function(x):
	"""A third order polynomial function
	"""

	return x ** 3 - x


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex10: Gaussian Processes',
						required = False)

	parser.add_argument('--n',
						default = 150,
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
