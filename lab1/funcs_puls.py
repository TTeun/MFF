import numpy as np
import math


def S(k,x,L):
	return np.sin(math.pi* (2*k+1)* x/L)

def Psi(k,t,sigma,omega):
	return (2*k+1)**2*sigma*math.pi**2*np.sin(omega*t) - omega*np.cos(omega*t) + omega*np.exp(-1*(2*k+1)**2*sigma*math.pi**2*t)

def sigma(mu, rho, L):
	return float(mu) / (rho * L * L)

def d_k(k, rho, sigma, omega):
	return 4. / (rho * math.pi * (2. * k + 1.)  * ( (2. * k + 1. ) ** 4 * sigma ** 2 * math.pi ** 4 + omega ** 2))
	
def cc(x):
	return np.cos(x) * np.cosh(x)

def ss(x):
	return np.sin(x) * np.sinh(x)
