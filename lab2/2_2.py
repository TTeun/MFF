from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

# supress uncecessary output
set_log_level(ERROR)


class T_TOP(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1.0))

class T_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0))

class T_LEFT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0))

class T_RIGHT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1.0))

# T_D = T_D()
# T_N = T_N()

# T_D.mark(boundaries, 0)
# T_N.mark(boundaries, 1)

def diffreac(N,  bc_type='dirichlet'):
	# Create mesh and define function space
	mesh = UnitSquareMesh(N, N)
	V = FunctionSpace(mesh, "Lagrange", 1)
	u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)

	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("-10", degree=2)

	if (bc_type == 'dirichlet'):
		boundaries = FacetFunction('size_t', mesh)
		bc = DirichletBC(V, u0, boundaries, 0)
		L = f*v*dx 

	a = inner(grad(u), grad(v))*dx
	u = Function(V)
	solve(a == L, u, bc)

	# Plot solution
	plot(u, interactive=True)
	plt.show()

	diff = (u0 - u) * (u0 -u);
	print errornorm(u0, u)


diffreac(10)