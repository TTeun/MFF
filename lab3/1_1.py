from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

def boundary(x, on_boundary):
    return on_boundary

def coth(x):
	return 1. / tanh(x)

def adv_dif_equation(mu = 0.1, N = 32, show_plot=False):
	mesh = UnitSquareMesh(N, N)

	#mu = max(mu, 2. / N)

	# const = Constant('exp(-2. / mu)')
	u0 = Expression('x[0] * ( (1 - exp( (x[1] - 1) / mu )) / (1 - exp(-2. / mu)))', mu=mu, degree = 3)
	b = Constant([0.,1.])

	V = FunctionSpace(mesh, "Lagrange", 1)
	u = TrialFunction(V)
	v = TestFunction(V)
	a = mu * inner(grad(u), grad(v))*dx + inner(b, grad(u) * v) * dx
	L = Expression('0', degree=1) * v * ds

	Pe = 1. / (N * 2. * mu)
	delta = (2. / N) * (coth(Pe) - 1. / Pe)
	S = delta * inner(b, grad(v)) * inner(b, grad(u)) * dx
	a += S

	bc = DirichletBC(V, u0, boundary)

	u_sol = Function(V)

	solve(a == L, u_sol, bc)
	if (show_plot):
		plot(u_sol)
		plt.title('sdaas')
		plt.show()

	return errornorm(u0, u_sol, 'H1', degree_rise=0)

def get_convergence_graph(mu):
	x = []
	y = []
	k_max = 7
	for k in range(k_max):
		if (k == k_max - 1):
		    y.append(adv_dif_equation(mu, 2 ** (k + 1), True))
		else:
			y.append(adv_dif_equation(mu, 2 ** (k + 1)))
		
		x.append(2 ** (k + 1))

	fig = plt.figure()
	plt.title('Convergence of solution to a(u,v) = L(v) for mu=%f' %mu)
	plt.loglog(x, y, basex=2, basey=2)
	plt.grid()
	plt.show(block=True)


# get_convergence_graph(0.1)
get_convergence_graph(1e-3)
get_convergence_graph(1e-6)


