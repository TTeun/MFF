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

def transient_Nstokes(finemesh = False, dt = 0.01, theta = 1., Re = 10., print_interval = 1):
	# Some constants
	mu = 0.035
	rho = 1.2
	Tf = 0.4
	R = 1
	u_bulk = Re * mu / (2. * rho * R)

	u_in = Expression(('3/2 * u_bulk * sin(pi * t / Tf) * (1 - (x[1] / R) * (x[1] / R))', '0'), u_bulk = u_bulk, t = 0., Tf = Tf, R = R, degree=3)


	# Create mesh
	if finemesh:
		mesh = Mesh('stenosis_f0.6_fine/stenosis_f0.6_fine.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6_fine/stenosis_f0.6_fine_facet_region.xml')
	else:
		mesh = Mesh('stenosis_f0.6/stenosis_f0.6.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6/stenosis_f0.6_facet_region.xml')
	
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# define elements and functionspace
	VE = VectorElement('P', mesh.ufl_cell(), 2)
	FE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * FE

	W = FunctionSpace(mesh, ME)
	W0 = FunctionSpace(mesh, VE)

	# Define functions
	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)

	# Set up the problem
	d = Expression("d * t", d = dt, t = theta, degree=1)
	d_minus = Expression("d * (1. - t)", d = dt, t = theta, degree=1)
	a = rho * inner( u , v) * dx + d * mu * inner(grad(u), grad(v)) * dx - dt * p * div(v) * dx + d * q * div(u) * dx

	u0 = interpolate(Constant([0,0]), W0)

	a += d * rho * inner( grad(u) * u0, v) * dx

	# Add Dirichlet bc's
	bcs = []
	bc = DirichletBC(W.sub(0), u_in, boundaries, 1)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 3)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0), [0,0], boundaries, 4)
	bcs.append(bc)

	A = assemble(a)
	[bc.apply(A) for bc in bcs]

	sol = XDMFFile('u3.xdmf')
	sol.parameters['rewrite_function_mesh'] = False

	index = 0
	for t in np.arange(dt, Tf + dt, dt):
		n = Constant([-1,0])
		u_in.t = t + (theta-1) * dt
		L = rho * inner( u0 , v) * dx - d_minus * mu * inner(grad(u0), grad(v)) * dx - d_minus * q * div(u0) * dx
		L -= d_minus * rho * inner( grad(u0) * u0, v) * dx

		assemble(a, tensor=A)
		b = assemble(L)
		[bc.apply(b) for bc in bcs]
		w = Function(W, name='solution')
		solve(A, w.vector(), b)
		u0, _ = split(w)
		
		index += 1
		if (index % print_interval == 0):
			print t
			sol.write(w.split()[0], t)
		
#transient_stokes(dt = 0.05, theta = 0.5)
transient_Nstokes()