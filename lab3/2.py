from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

def boundary(x, on_boundary):
    return on_boundary

def coth(x):
	return 1. / tanh(x)

def adv_dif_equation(mu = 0.1, N = 32, SUPG = False, show_plot=False):
	mesh = UnitSquareMesh(N, N)

	f = Expression(' (x[0] <= 0.1 && x[1] <= 0.1) ? 10 : 0', degree=2)
	b = Constant([1. / sqrt(2.),1. / sqrt(2.)])

	V = FunctionSpace(mesh, "Lagrange", 2)
	u = TrialFunction(V)
	v = TestFunction(V)
	a = mu * inner(grad(u), grad(v))*dx + inner(b, grad(u) * v) * dx
	L = f * v * ds

	if SUPG:
		Pe = 1. / (N * 2. * mu)
		delta = (2. / N) * (coth(Pe) - 1. / Pe)
		S = delta * inner(b, grad(v)) * inner(b, grad(u)) * dx
		a += S

	bc = DirichletBC(V, 0, boundary)
	u_sol = Function(V)
	solve(a == L, u_sol, bc)
	if (show_plot):
		plot(u_sol)
		plt.title('sdaas')
		plt.show()

adv_dif_equation(1e-3, 64, True, True)

