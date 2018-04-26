from dolfin import *
from numpy import *
import matplotlib.pyplot as plt


def get_approximation(N):
	# Create mesh and define function space
	mesh = UnitSquareMesh(N, N)
	V = FunctionSpace(mesh, "Lagrange", 1)

	# Define Dirichlet boundary (x = 0 or x = 1)
	def boundary(x):
	    return 	(	x[0] < DOLFIN_EPS or 
	    			x[0] > 1.0 - DOLFIN_EPS or 
	    			x[1] < DOLFIN_EPS or 
			    	x[1] > 1.0 - DOLFIN_EPS )

	# Define boundary condition
	u0 = Expression("x[0] * x[1] + cos(2 * pi * x[0]) * sin(2 * pi * x[1])", degree=4)
	f = Expression("8 * pi * pi * cos(2 * pi * x[0]) * sin(2 * pi * x[1]) ", degree=4)
	bc = DirichletBC(V, u0, boundary)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	a = inner(grad(u), grad(v))*dx
	L = f*v*dx 

	# Compute solution
	u = Function(V)
	solve(a == L, u, bc)

	diff = (u0 - u) * (u0 -u);
	return errornorm(u0, u)

x = []
y = []
for k in range(6):
	y.append(get_approximation(2 ** (k + 1)))
	x.append(2 ** (k + 1))

plt.ion()
fig = plt.figure()
plt.loglog(x,y, basex=2, basey=2)
plt.grid()

fig.savefig('error.png')

plt.show(block=True)