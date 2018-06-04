from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

Len = 6.

# Define domains
class T_LEFT(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and (near(x[0], 0.0))

class T_RIGHT(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and (near(x[0], Len) )
		
class T_TOP(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and (near(x[1], 1.0))

class T_BOTTOM(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and (near(x[1], 0.0))

def transient_stokes(dt, theta):
	# Some constants
	mu = 0.035
	ny = 16
	nx = 6*ny
	rho = 1.2
	Tf = 2

	# Neumann condition for x = 0
	f = Expression("4000. * sin(2 * pi * t) + 2000. * sin(4 * pi * t) + 1333.3 * sin(6 * pi * t)", t=0, degree=3)

	# Create mesh
	mesh = RectangleMesh(Point(0,0), Point(Len,1), nx, ny)

	# define elements and functionspace
	VE = FiniteElement('P', mesh.ufl_cell(), 2)
	W = FunctionSpace(mesh, VE)

	# Define functions
	u = TrialFunction(W)
	v = TestFunction(W)

	# Split up the boundary
	boundaries = FacetFunction('size_t', mesh)
	T_l, T_r, T_t, T_b = [T_LEFT(), T_RIGHT(), T_TOP(), T_BOTTOM()]
	T_l.mark(boundaries, 1)
	T_r.mark(boundaries, 2)
	T_t.mark(boundaries, 3)
	T_b.mark(boundaries, 4)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# Set up the problem
	d = Expression("d * t * m", d = dt, t = theta, m = mu, degree=1)
	a = rho * u * v * dx + d * inner(grad(u), grad(v)) * dx 

	u0 = interpolate(Constant(0), W)

	# Add Dirichlet bc's
	bcs = []
	bc  = DirichletBC(W, 0, boundaries, 3)
	bcs.append(bc)

	A = assemble(a)
	[bc.apply(A) for bc in bcs]

	sol = XDMFFile('u.xdmf')
	sol.parameters['rewrite_function_mesh'] = False

	index = 0
	for t in np.arange(dt, Tf + dt, dt):
		d_minus = Expression("d * (1. - t)", d = dt, t = theta, degree=1)
		L = rho * u0 * v * dx - d_minus * inner(grad(u0), grad(v)) * dx
		f.t = t + theta * dt
		L += dt * f / Len * v * dx

		b = assemble(L)
		u1 = Function(W, name='solution')
		solve(A, u1.vector(), b)
		++index
		if (index % 20 == 0):
			sol.write(u1, t)
		
		u0.assign(u1)

transient_stokes(0.0005, 0.0)