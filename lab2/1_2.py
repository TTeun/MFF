from dolfin import *
from numpy import *

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "Lagrange", 2)

class T_D(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0)) 

class T_N(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], 1.0)) 

T_D = T_D()
T_N = T_N()

boundaries = FacetFunction('size_t', mesh)
T_D.mark(boundaries, 1)
T_N.mark(boundaries, 2)


# Define boundary condition
u0 = Expression("1 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)
bc = DirichletBC(V, u0, boundaries, 1)
g = Expression("6 * x[1]", degree=1)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("-10", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds(2)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

File("poisson2.pvd") << u

# Plot solution
plot(u, interactive=True)

diff = (u0 - u) * (u0 -u);
print sqrt(assemble (diff * dx))
print errornorm(u0, u)