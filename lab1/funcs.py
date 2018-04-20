def average(values):
	return sum(values) / len(values)

def variance(values):
	m = average(values)
	var = 0.
	for x in (values):
		var += (x - m)**2

	return var

def covarience(values_x, values_y):
	m_x = average(values_x)
	m_y = average(values_y)
	covar = 0.
	for x, y in zip(values_x, values_y):
		covar += (x - m_x) * (y - m_y)

	return covar


def alpha(values_x, values_y):
	return covarience(values_x, values_y) / variance(values_x)