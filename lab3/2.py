from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

# boundaries
class T_LEFT_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[1], 0.0))
		
class T_RIGHT_TOP(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1.0) or near(x[1], 1.0))


def coth(x):
	return 1. / tanh(x)

def adv_dif_equation(mu = 0.1, N = 32, SUPG = False, show_plot = False, Neum = False):

	# create mesh
	mesh = UnitSquareMesh(N, N)
	
	# f and b
	f = Expression(' (x[0] <= 0.1 && x[1] <= 0.1) ? 10. : 0.', degree=2)
	b = Constant([1. / sqrt(2.),1. / sqrt(2.)])

	# define function space
	V = FunctionSpace(mesh, "Lagrange", 2)
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
	if not Neum: # apply Dirichlet bc on whole boundary
		bcs.append(DirichletBC(V, 0, boundaries, 2))
		 
	# Solve the shit
	u_sol = Function(V)
	solve(a == L, u_sol, bcs)
    
	print 'minimum of u =' , min(u_sol.vector().array()) # if <0 then wiggles
	
	if (show_plot): # plot the solution
		plot(u_sol)
		plt.title('sdaas')
		plt.show()
		File('sol.pvd') << u_sol

#adv_dif_equation(1e-3, 32, False, True)
adv_dif_equation(1e-3, 32, True, True, True)

