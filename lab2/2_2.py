from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

# supress uncecessary output
set_log_level(ERROR)


class T_HORI(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1.0) or near(x[1], 0.0))

class T_VERT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0))

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
	V = FunctionSpace(mesh, "Lagrange", 2)
	u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)

	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("-10", degree=2)
	a = inner(grad(u), grad(v))*dx
	u = Function(V)
	
	boundaries = FacetFunction('size_t', mesh)
	if (bc_type == 'dirichlet'):
		bc = DirichletBC(V, u0, boundaries, 0)
		L = f*v*dx 
		solve(a == L, u, bc)

	if (bc_type == 'neumann'):
		u_y = Expression('6 * x[1]', degree=2)
		u_x = Expression('4 * x[0]', degree=2)
		T_h = T_HORI()
		T_v = T_VERT()
		T_h.mark(boundaries, 0)
		T_v.mark(boundaries, 1)
		L = f*v*dx + u_y * v * ds(0) + u_x * v * ds(1)
		solve(a == L, u)

	# Plot solution
	plot(u, interactive=True)
	plt.show()

	diff = (u0 - u) * (u0 -u);
	print errornorm(u0, u)


diffreac(16, 'dirichlet')