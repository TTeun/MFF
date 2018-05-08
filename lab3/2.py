from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

set_log_level(CRITICAL)

# boundaries
class T_LEFT_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[1], 0.0))
		
class T_RIGHT_TOP(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1.0) or near(x[1], 1.0))

# For the supg calculation
def coth(x):
	return 1. / tanh(x)

def adv_dif_equation(mu = 0.1, N = 32, SUPG = False, show_plot = False, Neum = False, JustPeclet = False):
	# create mesh
	mesh = UnitSquareMesh(N, N)
	
	# f and b
	f = Expression('(x[0] <= 0.1 && x[1] <= 0.1) ? 10. : 0.', degree=2)
	b = Constant([1. / sqrt(2.),1. / sqrt(2.)])

	# define function space
	V = FunctionSpace(mesh, "Lagrange", 1)
	u = TrialFunction(V)
	v = TestFunction(V)
	
	# The weak form
	a = mu * inner(grad(u), grad(v))*dx + inner(b, grad(u) * v) * dx
	L = f * v * dx
	
	# Split up the boundary
	boundaries = FacetFunction('size_t', mesh)
	T_lb, T_rt = [T_LEFT_BOTTOM(), T_RIGHT_TOP()]
	T_lb.mark(boundaries, 1)
	T_rt.mark(boundaries, 2)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	
	# compute local Peclet number
	infnorm = 1. / sqrt(2.)
	Pe = infnorm / (2. * N * mu)
	if JustPeclet:
		return Pe

	print 'Pe =',  Pe

	if SUPG: # use SUPG
		delta = 2. / (N * infnorm) * (coth(Pe) - 1. / Pe)
		
		# stabilization term
		S = delta * inner(b, grad(v)) * inner(b, grad(u)) * dx
		
		# add terms to problem formulation
		a += S
		L += delta * f * inner(b, grad(v)) * dx

	# set up boundary conditions
	bcs = [DirichletBC(V, 0, boundaries, 1)]

	# apply Dirichlet bc on whole boundary
	if not Neum: 
		bcs.append(DirichletBC(V, 0, boundaries, 2))
		 
	# Solve the system!
	u_sol = Function(V)
	solve(a == L, u_sol, bcs)
    
	print 'minimum of u =' , min(u_sol.vector().array()) # if <0 then wiggles
	
	# plot the solution
	if (show_plot): 
		plot(u_sol)
		plt.title('Density')
		plt.show()
		File('sol.pvd') << u_sol

# Convenience functions
def show_neumann_with_supg(N):
	adv_dif_equation(1e-3, N, True, True, True)

def show_neumann(N):
	adv_dif_equation(1e-3, N, False, True, True)

def show_dirichlet(N):
	adv_dif_equation(1e-3, N, False, True, False)

def show_dirichlet_with_supg(N):
	adv_dif_equation(1e-3, N, True, True, False)

# Print the Peclet numbers for different mesh sizes
def show_peclet_convergence(low, high):
	for k in range(low, high):
		print adv_dif_equation(1e-3, 2 ** k, False, True, False, True)

# show_dirichlet(64)
# show_dirichlet_with_supg(64)
show_peclet_convergence(5, 10)