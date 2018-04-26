'''

This file consist of all the functions used for the 'Pulsating channel flow' assignment

'''

import numpy as np
import math

 # sigma
def sigma(mu, rho, L):
	return float(mu) / (rho * L * L)

# d_k
def d_k(k, rho, sigma, omega):
	return 4. / (rho * math.pi * (2. * k + 1.)  * ( (2. * k + 1. ) ** 4 * sigma ** 2 * math.pi ** 4 + omega ** 2))

def s_k(k,x,L): # the S_k(x) function
	x2 = [x_i * math.pi * (2. * k + 1.) / L for x_i in x]
	return np.sin(x2)

# the Psi_k(t) function
def psi_k(k,t,sigma,omega): 
	c1 = (2. * k + 1.) ** 2
	result = c1 * sigma * math.pi ** 2 * np.sin(omega * t)
	result -= omega * np.cos(omega * t) 
	result += omega * np.exp(-1. * c1 * sigma * math.pi ** 2 * t)
	return  result

# computes the velocity vector of the transient flow at time t and initial u = 0
def transient(t, max_k = 20, mu = 0.035, rho = 1., L = 1., a = 1., omega = 1.): 
	s = sigma(mu, rho, L)
	x = list(np.linspace(0, L))
	u = [0.] * len(x)
	for k in range(max_k):
		u += a * d_k(k, rho, s, omega) * psi_k(k, t, s, omega) * s_k(k, x, L)
	return u

# the cc(x) function
def cc(x): 
	return np.cos(x) * np.cosh(x)

# the ss(x) function
def ss(x):
	return np.sin(x) * np.sinh(x)
	
def kappa(mu,rho,omega):
	return (omega * rho / (2 * mu)) ** 0.5

# computes the f_1(x), f_2(x) and f_3(x) functions.
def f1_f2_f3(x,mu,rho,L,omega): 
	cc1 = [cc(kappa(mu, rho, omega) * (x_i - L/2.)) for x_i in x]
	cc2 = cc(kappa(mu, rho, omega) * L/2.)
	ss1 = [ss(kappa(mu, rho, omega) * (x_i - L/2.)) for x_i in x]
	ss2 = ss(kappa(mu, rho, omega) * L/2.)
	
	f1 = [cc1_i * cc2 + ss1_i * ss2 for cc1_i, ss1_i in zip(cc1,ss1)]
	f2 = [cc1_i * ss2 - ss1_i * cc2 for cc1_i, ss1_i in zip(cc1,ss1)]
	f3 = cc2**2 + ss2**2
	return f1,f2,f3

# computes the velocity vector of the periodic flow at time t and initial u = 0
def periodic(t, mu = 0.035, rho = 1., L = 1., a = 1., omega = 1.): 
	x = list(np.linspace(0, L))
	f1,f2,f3 = f1_f2_f3(x,mu,rho,L,omega)
	u = [-a/omega * (f2_i/f3 * np.sin(omega * t) - (1 - f1_i/f3) * np.cos(omega * t)) for f1_i,f2_i in zip(f1,f2)]
	return u	
