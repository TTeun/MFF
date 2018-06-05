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

def transient_stokes(epsilon, P2P1):
	# Some constants
	mu = 0.035
	ny = 16
	nx = 6*ny
	rho = 1.2
	Tf = 2
	dt = 0.01
	theta = 1

	f = Expression("4000. * sin(2 * pi * t) + 2000. * sin(4 * pi * t) + 1333.3 * sin(6 * pi * t)", t=0, degree=3)


	# Create mesh
	mesh = RectangleMesh(Point(0,0), Point(Len,1), nx, ny)

	# define elements and functionspace
	if P2P1:
		VE = VectorElement('P', mesh.ufl_cell(), 2)
	else:
		VE = VectorElement('P', mesh.ufl_cell(), 1)
	FE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * FE

	W = FunctionSpace(mesh, ME)
	W0 = FunctionSpace(mesh, VE)

	k = W.sub(0).ufl_element().degree()
	# function space of kinetic energy
	V_KE = FunctionSpace(mesh, 'P', k**2)

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

	if not P2P1:
		h = CellDiameter(mesh)
		a += epsilon * h * h / mu * inner(grad(p), grad(q)) * dx

	u0 = interpolate(Constant([0,0]), W0)

	# Add Dirichlet bc's
	bcs = []
	bc = DirichletBC(W.sub(0), [0,0], boundaries, 3)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 4)
	bcs.append(bc)

	A = assemble(a)
	[bc.apply(A) for bc in bcs]

	solu = XDMFFile('u4.xdmf')
	solu.parameters['rewrite_function_mesh'] = False

	solp = XDMFFile('p4.xdmf')
	solp.parameters['rewrite_function_mesh'] = False

	index = 0
	e_kin = []
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

		e_kinfun = project(dot(u0, u0), V_KE)
		e_kin.append(assemble(e_kinfun * dx))

		if (index % 10 == 0):
			print t
			solu.write(w1.split()[0], t)
			solp.write(w1.split()[1], t)
		
	x = np.arange(dt, Tf + dt, dt)
	fig = plt.figure()
	plt.title('The flow kinetic energy')
	plt.plot(x, e_kin)
	plt.show()

transient_stokes(5., False)