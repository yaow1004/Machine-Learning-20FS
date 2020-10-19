# Imports

import argparse
import random
import time
import numpy as np
import matplotlib
from scipy.stats import multivariate_normal

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def init():

	# Variables

	n = args.n
	sigma = 0.5 ** 2
	tau = 5.0 ** 2

	# Sampled observations

	x = []
	y = []

	# Init plot

	ax = plt.gca()

	# Sampling loop

	for i in range(n):
		# Clear previous plot

		plt.cla()

		# Add a new sample to the observations

		x_, y_ = data_generator(1, third_order_function, sigma)
		x.append(x_[0])
		y.append(y_[0])

		# Model selection

		d = model_selection(x, y, sigma, tau)
		d = 5
		print('Selected polynomial order: {}'.format(d))

		# Compute posterior

		beta_hat, covar = posterior(poly_expansion(x, d), y, sigma, tau)

		# Evaluate regression function

		px = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution for i in range(args.plot_resolution)]
		pxx = poly_expansion(px, d)
		py, py_std = posterior_predictive(pxx, beta_hat, covar, sigma)

		# Uncertainty boundaries

		upper = (py + py_std).flatten().tolist()[0]
		lower = (py - py_std).flatten().tolist()[0]

		# Plot

		plt.axis(args.plot_boundaries, fontsize = args.font_size)
		plt.fill_between(px, upper, lower, alpha = 0.5, label = 'Standard deviation')
		plt.scatter(x, y, s = args.scatter_size, label = 'Observations')
		plt.plot(px, py, color = 'red', label = 'Polynomial estimate')

		ax.legend(loc = 'upper right')
		plt.rc('xtick', labelsize = args.font_size)
		plt.rc('ytick', labelsize = args.font_size)
		plt.title(args.title)
		plt.xlabel('X', fontsize = args.font_size)
		plt.ylabel('Y', fontsize = args.font_size)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.1)

	plt.show()


def third_order_function(x):
	"""A third order polynomial function
	"""

	return x ** 3 - x

def poly_expansion(x, d):
	"""Polynomial expansion of vector x
	"""

	xx = [[pow(xi, j) for j in range(0, d)] for xi in x]

	return np.array(xx)


def polynomial_value(x, beta):
	"""Evaluate polynomial at data X, given weight parameters beta
	"""

	return [sum([b * (xi ** j) for j, b in enumerate(beta)]) for i, xi in enumerate(x)]


def data_generator(n, f, sigma):
	"""Generate training data
	"""

	x = [3 * (random.random() - 0.5) for _ in range(n)]
	y = [np.random.normal(f(xi), sigma ** 0.5, 1)[0] for xi in x]

	return x, y


def posterior(X, y, sigma, tau):
	"""Posterior
	"""

	# TODO: Compute the mean vector (beta estimate)
	_, n = X.shape
	I = np.eye(n)

	beta = np.linalg.inv(X.T @ X + (sigma / tau) * I) @ X.T @ y

	# TODO: Compute the covariance matrix

	covar = np.linalg.inv(1/sigma * (X.T @ X) + 1/tau * I)

	print(f'beta: {beta}, covar: {covar}')

	return beta, covar


def posterior_predictive(X, beta_hat, covar, sigma):
	"""Posterior predictive
	"""

	# TODO: Compute the y prediction given model parameters beta
	X=np.array(X)
	y = []

	for i in range(100):
		mu = (X[i].T @ beta_hat)
		cov = sigma + X[i].T @ covar @ X[i]
		y.append(multivariate_normal.pdf(X[i][1], mean=mu, cov=cov))


	# TODO: Compute the standard deviation
	y_std = np.sqrt(cov)
	return y, y_std

def model_selection(x, y, sigma, tau):
	"""Model selection
	"""

	n = len(y)
	d_min = 1
	d_max = 20
	score_list = []

	for d in range(d_min, d_max):
		xx = poly_expansion(x, d)

		# TODO: Predict y given x. Use the posterior and posterior_predictive methods.
		#x=np.array(x)
		#_, n = x.shape
		#I=np.eye(n)
		#beta_hat = np.linalg.inv(x.T@x +sigma/tau*I)@x.T@y
		#mu = x.T @ beta_hat
		#cov = sigma
		#py = multivariate_normal.pdf(xx, mean= mu, cov =cov)
		py = 0

		# TODO: Compute the log likelihood, use the multivariate_normal method

		#log_likelihood = multivariate_normal.logpdf(xx, mean= mu, cov =cov)
		log_likelihood = 0

		# TODO: Compute the BIC score

		#score = np.sum(log_likelihood) - dof(beta_hat)/2*np.log(n)
		score = 0
		score_list.append(score)

	# TODO: Select a proper degree of freedom: k + d_min

	d =  d_min

	return d


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex5: Bayesian Linear Regression',
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
