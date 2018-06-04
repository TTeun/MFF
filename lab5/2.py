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
deltat = 0.1
theta = 0.5

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
a = rho * u * v * dx + deltat * theta * mu * inner(grad(u), grad(v)) * dx 

u0 = interpolate(Constant(0), W)

# Add Dirichlet bc's
bcs = []
bc = DirichletBC(W, 0, boundaries, 3)
bcs.append(bc)

A = assemble(a)
[bc.apply(A) for bc in bcs]

sol = XDMFFile('u.xdmf')
sol.parameters['rewrite_function_mesh'] = False

for t in np.arange(0.0, Tf, deltat):
	n = Constant([1,0])
	L = rho * u0 * v * dx - deltat * (1 - theta) * mu * inner(grad(u0), grad(v)) * dx
	L -=  func(t + theta * deltat) / Len * v * dx

	b = assemble(L)
	u1 = Function(W, name='solution')
	solve(A, u1.vector(), b)
	sol.write(u1, t)
	
	u0.assign(u1)
