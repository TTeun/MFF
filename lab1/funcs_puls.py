import numpy as np
import math
		

def sigma(mu, rho, L):
	return float(mu) / (rho * L * L)

def d_k(k, rho, sigma, omega):
	return 4. / (rho * math.pi * (2. * k + 1.)  * ( (2. * k + 1. ) ** 4 * sigma ** 2 * math.pi ** 4 + omega ** 2))
	
def cc(x):
	return np.cos(x) * np.cosh(x)

def ss(x):
	return np.sin(x) * np.sinh(x)
