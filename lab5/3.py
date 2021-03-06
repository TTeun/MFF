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

def transient_stokes(dt, theta, print_interval = 1):
	# Some constants
	mu = 0.035
	ny = 16
	nx = 6*ny
	rho = 1.2
	Tf = 2

	f = Expression("4000. * sin(2 * pi * t) + 2000. * sin(4 * pi * t) + 1333.3 * sin(6 * pi * t)", t=0, degree=3)


	# Create mesh
	mesh = RectangleMesh(Point(0,0), Point(Len,1), nx, ny)

	# define elements and functionspace
	VE = VectorElement('P', mesh.ufl_cell(), 2)
	FE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * FE

	W = FunctionSpace(mesh, ME)
	W0 = FunctionSpace(mesh, VE)

	# Define functions
	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)

	# Split up the boundary
	boundaries = FacetFunction('size_t', mesh)
	T_l, T_r, T_t, T_b = [T_LEFT(), T_RIGHT(), T_TOP(), T_BOTTOM()]
	T_l.mark(boundaries, 1)
	T_r.mark(boundaries, 2)
	T_t.mark(boundaries, 3)
	T_b.mark(boundaries, 4)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


	# Set up the problem
	d = Expression("d * t", d = dt, t = theta, degree=1)
	d_minus = Expression("d * (1. - t)", d = dt, t = theta, degree=1)
	a = rho * inner( u , v) * dx + d * mu * inner(grad(u), grad(v)) * dx - dt * p * div(v) * dx + d * q * div(u) * dx

	u0 = interpolate(Constant([0,0]), W0)

	# Add Dirichlet bc's
	bcs = []
	bc = DirichletBC(W.sub(0), [0,0], boundaries, 3)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 4)
	bcs.append(bc)

	A = assemble(a)
	[bc.apply(A) for bc in bcs]

	sol = XDMFFile('u3.xdmf')
	sol.parameters['rewrite_function_mesh'] = False

	index = 0
	for t in np.arange(dt, Tf + dt, dt):
		n = Constant([-1,0])
		L = rho * inner( u0 , v) * dx - d_minus * mu * inner(grad(u0), grad(v)) * dx - d_minus * q * div(u0) * dx
		f.t = t + (theta-1) * dt
		L -= dt * f * inner(n,v) * ds(1)

		b = assemble(L)
		[bc.apply(b) for bc in bcs]
		w1 = Function(W, name='solution')
		solve(A, w1.vector(), b)
		u0, _ = split(w1)
		
		index += 1
		if (index % print_interval == 0):
			print t
			sol.write(w1.split()[0], t)
		
#transient_stokes(dt = 0.05, theta = 0.5)
transient_stokes(dt = 0.05, theta = 1.)