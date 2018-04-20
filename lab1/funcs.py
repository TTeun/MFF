from random import gauss
import numpy as np

def average(values):
	return sum(values) / len(values)

def N_times_variance(values):
	m = average(values)
	var = 0.
	for x in (values):
		var += (x - m)**2

	return var

def N_times_covarience(values_x, values_y):
	m_x = average(values_x)
	m_y = average(values_y)
	covar = 0.
	for x, y in zip(values_x, values_y):
		covar += (x - m_x) * (y - m_y)

	return covar


def alpha(values_x, values_y):
	return N_times_covarience(values_x, values_y) / N_times_variance(values_x)

def beta(values_x, values_y):
	return average(values_y) - alpha(values_x, values_y) * average(values_x)

def alpha_and_beta(values_x, values_y):
	a = alpha(values_x, values_y)
	b = average(values_y) - a * average(values_x)
	return a, b


def x_and_y(N):
	x = [float(i) / N for i in range(N + 1)]
	g = gauss(0., 1.)
	y = [x_i + gauss(0., 1.) for x_i in x]
	return x, y

def alpha_and_beta_numpy(values_x, values_y):
	return np.polyfit(values_x, values_y,1)