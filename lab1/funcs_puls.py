import numpy as np
import math


def s_k(k,x,L):
	x[:] = [x_i * math.pi * (2. * k + 1.) / L for x_i in x]
	return np.sin(x)

def psi_k(k,t,sigma,omega):
	c1 = (2 * k + 1) ** 2
	result = c1 * sigma * math.pi ** 2 * np.sin(omega * t)
	result -= omega * np.cos(omega * t) 
	result += omega * np.exp(-1 * c1 * sigma * math.pi ** 2*t)
	return  result

def sigma(mu, rho, L):
	return float(mu) / (rho * L * L)

def d_k(k, rho, sigma, omega):
	return 4. / (rho * math.pi * (2. * k + 1.)  * ( (2. * k + 1. ) ** 4 * sigma ** 2 * math.pi ** 4 + omega ** 2))

def transient(t, max_k, mu = 0.035, rho = 1, L = 1, a = 1, omega = 1):
	s = sigma(mu, rho, L)
	x = list(np.linspace(0, L))
	u = [0.] * len(x)
	for k in range(max_k):
		u += a * d_k(k, rho, s, omega) * psi_k(k, t, s, omega) * s_k(k, x, L)
	return u


print (transient(1, 15) - transient(1, 20))



def cc(x):
	return np.cos(x) * np.cosh(x)

def ss(x):
	return np.sin(x) * np.sinh(x)
