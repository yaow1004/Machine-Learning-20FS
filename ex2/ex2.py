# Imports

import argparse
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class LCG:
	"""Linear congruential generator
	"""

	def __init__(self, a, c, m):

		# TODO: Set the initial random number to a proper random seed value of your choice.

		self.rand_number = 10
		self.a = a
		self.c = c
		self.m = m

	def draw_rand_number(self):

		# TODO: Implement the LCG

		self.rand_number = (self.a * self.rand_number + self.c) % self.m
        

		# TODO: Return a normalized random number x ~ U(0, 1)
		return self.rand_number / self.m
    

def rand_uniform(a, b):
	"""Draw a random number from an uniform distribution U(a, b)
	"""

	# TODO: Return a random number x ~ U(a, b). Use the linear congruential generator.

	return lcg.draw_rand_number() * (b - a) + a


def pareto_pdf(x, start_x, shape):
    b = start_x
    K = shape

#	"""Pareto probability density function

#	"""

	# TODO: Implement the Pareto probability density function.

    if (x<b):
        pdf = 0
    else:
        pdf = (K*b**K) / (x**(K+1))

    return pdf
            


def init():

	# The true distribution of the population U(0, theta_pop), where theta_pop is unknown

	theta_pop = args.theta_pop

	# The prior distribution Pa(theta ; theta_prior, shape_prior)

	theta_prior = args.theta_prior
	shape_prior = args.shape_prior

	# Sampled observations
	samples_pop = []  #observations x = {x1, x2, . . . , xn} being identically distributed with Unif(0, θ)

	# Init plot

	ax = plt.gca()

	# Sampling loop

	for i in range(25):
		# Clear previous plot

		plt.cla()

		# Add a new sample to the observations

		samples_pop.append(rand_uniform(0, theta_pop))
        # Compute posterior parametrization
        # P(theta|x) = Pa(theta|c,N+K)
        # c = max(theta_hat, b)
        # N: the sample size
        # TODO: Implement the Posterior parameters
		theta_hat = max(samples_pop) 	
		b = theta_prior
		c = max([theta_hat,b])
		theta_post = c 
		N = len(samples_pop) 	
		K = shape_prior
		shape_post = N + K
       

		# Plot samples from the population

		ax.hist(samples_pop, density = True, label = 'Samples from population')

		# Plot posterior

		x = [args.plot_boundaries[0] + i * (args.plot_boundaries[1] - args.plot_boundaries[0]) / args.plot_resolution for i in range(args.plot_resolution)]
		y = [pareto_pdf(xi, theta_post, shape_post) for xi in x]
		ax.plot(x, y, label = 'Posterior Pa(theta | c, N + K)')

		# Redraw

		ax.legend(loc='upper right')
		plt.axis(args.plot_boundaries, fontsize = args.font_size)
		plt.title(args.title)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.5)

	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex2: Bayesian analysis of the Uniform Distribution',
						required = False)

	parser.add_argument('--theta-pop',
						default = 500,
						required = False)

	parser.add_argument('--theta-prior',
						default = 0,
						required = False)

	parser.add_argument('--shape-prior',
						default = 0,
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [-5, 600, 0, 0.05],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	# LCG

	lcg = LCG(1664525, 1013904223, 2 ** 32)


	init()