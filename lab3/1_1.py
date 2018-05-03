from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

def boundary(x, on_boundary):
    return on_boundary



def adv_dif_equation(mu = 0.1, N = 32):
	mesh = UnitSquareMesh(N, N)

	# const = Constant('exp(-2. / mu)')
	u0 = Expression('x[0] * ( (1 - exp( (x[1] - 1) / mu )) / (1 - exp(-2. / mu)))', mu=mu, degree = 3)
	b = Constant([0,1])

	V = FunctionSpace(mesh, "Lagrange", 1)
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("-10", degree=2)
	a = mu * inner(grad(u), grad(v))*dx + inner(b, grad(u) * v) * dx
	L = Expression('0', degree=1) * v * ds

	bc = DirichletBC(V, u0, boundary)

	u_sol = Function(V)

	solve(a == L, u_sol, bc)
    
	# plot(u_sol)
	# plt.title('sdaas')
	# plt.show()

	return errornorm(u0, u_sol, 'H1', degree_rise=0)

def get_convergence_graph(mu):
	x = []
	y = []
	for k in range(7):
	    y.append(adv_dif_equation(mu, 2 ** (k + 1)))
	    x.append(2 ** (k + 1))

	fig = plt.figure()
	plt.title('Convergence of solution to a(u,v) = L(v) for mu=%f' %mu)
	plt.loglog(x, y, basex=2, basey=2)
	plt.grid()

	fig.savefig('error.png')

	plt.show(block=True)

get_convergence_graph(0.1)
get_convergence_graph(1e-3)
get_convergence_graph(1e-6)


