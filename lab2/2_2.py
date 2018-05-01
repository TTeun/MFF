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

def diffreac(N,  bc_type='dirichlet'):
	# Create mesh and define function space
	mesh = UnitSquareMesh(N, N)
	V = FunctionSpace(mesh, "Lagrange", 2)
	u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)

	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("-10", degree=2) + u0
	boundaries = FacetFunction('size_t', mesh)

	if (bc_type == 'dirichlet'):
		a = (u * v) * dx + inner(grad(u), grad(v))*dx
		u = Function(V)
		L = f*v*dx 
		bc = DirichletBC(V, u0, boundaries, 0)
		solve(a == L, u, bc)

	if (bc_type == 'neumann'):
		a = (u * v) * dx + inner(grad(u), grad(v))*dx
		u = Function(V)
		u_x = Expression('4 * x[0]', degree=1)
		u_y = Expression('6 * x[1]', degree=1)
		T_v = T_VERT()
		T_h = T_HORI()
		T_v.mark(boundaries, 0)
		T_h.mark(boundaries, 1)
		# bc = DirichletBC(V, u0, boundaries, 0)
		L += f*v*dx + u_y * v * ds(1) +  u_x * v * ds(0) 
		solve(a == L, u)

	if (bc_type == 'robin'):
		a = (u * v) * dx + inner(grad(u), grad(v))*dx - (u * v) * ds
		u = Function(V)
		L = f*v*dx 
		boundaries = FacetFunction('size_t', mesh)

		u_x = Expression('4 * x[0]', degree=1) + u0
		u_y = Expression('6 * x[1]', degree=1) + u0
		T_v = T_VERT()
		T_h = T_HORI()
		T_v.mark(boundaries, 0)
		T_h.mark(boundaries, 1)
		L += u_y * v * ds(1) +  u_x * v * ds(0) 
		solve(a == L, u)

	# Plot solution
	plot(u, interactive=True)
	plt.show()
	print errornorm(u0, u)

diffreac(16, 'robin')
# diffreac(16, 'neumann')
# diffreac(16, 'dirichlet')


		# u_top = Expression('6', degree=2)
		# u_bottom = Expression('0', degree=2)
		# u_left = Expression('0', degree=2)
		# u_right = Expression('4', degree=2)

		# T_top = T_TOP()
		# T_bottom = T_BOTTOM()
		# T_left = T_LEFT()
		# T_right = T_RIGHT()

		# T_top.mark(boundaries, 0)
		# T_bottom.mark(boundaries, 1)
		# T_left.mark(boundaries, 2)
		# T_right.mark(boundaries, 3)
		
		# L = f * v * ds +  v * u_top * ds(0) + v * u_bottom * ds(1) + v * u_left * ds(2) + v * u_right * ds(3) 
		# solve(a == L, u)


