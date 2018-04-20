from random import gauss
import numpy as np

def average(values):
	return sum(values) / len(values)

def in_product(values1, values2):
	result = 0.
	for x, y in zip(values1, values2):
		result += x * y
	return result

def alpha_and_beta(values_x, values_y):
	m_x = average(values_x)
	values_x[:] = [x_i - m_x for x_i in values_x]

	m_y = average(values_y)
	values_y[:] = [y_i - m_y for y_i in values_y]

	a = in_product(values_y, values_x) / in_product(values_x, values_x)
	b = m_y - a * m_x
	return a, b


def x_and_y(N):
	x = [float(i) / N for i in range(N + 1)]
	g = gauss(0., 1.)
	y = [x_i + gauss(0., 1.) for x_i in x]
	return x, y

def alpha_and_beta_numpy(values_x, values_y):
	return np.polyfit(values_x, values_y,1)