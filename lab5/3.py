from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

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

def func(t):
	f = 0.
	for j in range(1,4):
		f += (4000./j) * np.sin(j * 2 * np.pi * t)
	return f

# Some constants
Len = 6.
mu = 0.035
ny = 32
nx = 3*ny
ny = ny/2
rho = 1.2
Tf = 2
dt = 0.5
theta = 0.

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

for t in np.arange(0.0, Tf, dt):
	n = Constant([-1,0])
	L = rho * inner( u0 , v) * dx - d_minus * mu * inner(grad(u0), grad(v)) * dx - d_minus * q * div(u0) * dx
	L -= dt * func(t + theta * dt) * inner(n,v) * ds(1)

	b = assemble(L)
	w1 = Function(W, name='solution')
	solve(A, w1.vector(), b)
	u0, _ = split(w1)
	
	sol.write(w1.split()[0], t)
	
