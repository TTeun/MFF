from dolfin import *
from numpy import *
import matplotlib.pyplot as plt
from domains import *

# supress uncecessary output
set_log_level(ERROR)

def diffreac(N,  bc_type='dirichlet'):
	# Create mesh and define function space
	mesh = UnitSquareMesh(N, N)
	V = FunctionSpace(mesh, "Lagrange", 2)
	u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)

	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("-10", degree=2) + u0
	L = f*v*dx
	boundaries = FacetFunction('size_t', mesh)

	if (bc_type == 'dirichlet'):
		bc = DirichletBC(V, u0, boundaries, 0)

		a = (u * v) * dx + inner(grad(u), grad(v))*dx
		u = Function(V)

		solve(a == L, u, bc)

	if (bc_type == 'neumann'):
		T_v, T_h = [T_VERT(), T_HORI()]
		T_v.mark(boundaries, 0)
		T_h.mark(boundaries, 1)
 		ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

		u_x = Expression('4 * x[0]', degree=1)
		u_y = Expression('6 * x[1]', degree=1)

		a = (u * v) * dx + inner(grad(u), grad(v))*dx
		u = Function(V)
		L += u_y * v * ds(1) +  u_x * v * ds(0) 

		solve(a == L, u)

	if (bc_type == 'robin'):
		T_v, T_h = [T_VERT(), T_HORI()]
		T_v.mark(boundaries, 0)
		T_h.mark(boundaries, 1)
 		ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

		u_x = Expression('4 * x[0]', degree=1) + u0
		u_y = Expression('6 * x[1]', degree=1) + u0

		a = (u * v) * dx + inner(grad(u), grad(v))*dx - (u * v) * ds
		u = Function(V)
		L += u_y * v * ds(1) +  u_x * v * ds(0) 

		solve(a == L, u)

	if (bc_type == 'mixed'):
		T_left, T_right, T_hori = [T_LEFT(), T_RIGHT(), T_HORI()]
		T_left.mark(boundaries, 0)
		T_right.mark(boundaries, 1)
		T_hori.mark(boundaries, 2)
 		ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
		
		u_right = Expression('4', degree=1)
		u_y = Expression('6 * x[1]', degree=1) + u0
		bc = DirichletBC(V, u0, boundaries, 0)

		a = (u * v) * dx + inner(grad(u), grad(v))*dx - (u * v) * ds(2)
		u = Function(V)
		L += u_right * v * ds(1) + u_y * v * ds(2)

		solve(a == L, u, bc)

	# Plot solution
	plot(u, interactive=True)
	plt.title(bc_type)
	plt.show()
	print errornorm(u0, u)
	file = File(bc_type + '.pvd')
	file << u

diffreac(64, 'dirichlet')
diffreac(64, 'neumann')
diffreac(64, 'robin')
diffreac(64, 'mixed')