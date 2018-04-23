'''

This file consist of all the functions used for the 'Linear regression' assignment

'''

from random import gauss
import numpy as np

def average(values): # computes the mean of a list.
	return sum(values) / len(values)

def in_product(values1, values2): # computes the in-product between two lists
	result = 0.
	for x, y in zip(values1, values2):
		result += x * y
	return result

def alpha_and_beta(values_x, values_y): # compute the alpha and beta of the linear fit y = alpha x + beta
	m_x = average(values_x)
	values_x2 = [x_i - m_x for x_i in values_x]

	m_y = average(values_y)
	values_y2 = [y_i - m_y for y_i in values_y]

	a = in_product(values_y2, values_x2) / in_product(values_x2, values_x2)
	b = m_y - a * m_x

	return a, b


def x_and_y(N): # generate the x and y data
	x = [float(i) / N for i in range(N + 1)]
	y = [x_i + gauss(0., 1.) for x_i in x]
	return x, y

def alpha_and_beta_numpy(values_x, values_y): # compute alpha and beta with numpy package
	return np.polyfit(values_x, values_y, 1)

