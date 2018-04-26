from dolfin import *
from numpy import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return 	(	x[0] < DOLFIN_EPS or 
    			x[0] > 1.0 - DOLFIN_EPS or 
    			x[1] < DOLFIN_EPS or 
		    	x[1] > 1.0 - DOLFIN_EPS )

# Define boundary condition
u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("-10", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx 

# Compute solution
u = Function(V)
solve(a == L, u, bc)

File("poisson1.pvd") << u

# Plot solution
# plot(u, interactive=True)

diff = (u0 - u) * (u0 -u);
print sqrt(assemble (diff * dx))
print errornorm(u0, u)