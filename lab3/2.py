from dolfin import *
from numpy import *
import matplotlib.pyplot as plt

def boundary(x, on_boundary):
    return on_boundary


class T_LEFT_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[1], 0.0))
		
class T_RIGHT_TOP(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1.0) or near(x[1], 1.0))

		
	
def coth(x):
	return 1. / tanh(x)

def adv_dif_equation(mu = 0.1, N = 32, SUPG = False, show_plot = False, Neum = False):
	mesh = UnitSquareMesh(N, N)

	f = Expression(' (x[0] <= 0.1 && x[1] <= 0.1) ? 10. : 0.', degree=2)
	b = Constant([1. / sqrt(2.),1. / sqrt(2.)])

	V = FunctionSpace(mesh, "Lagrange", 2)
	u = TrialFunction(V)
	v = TestFunction(V)
	a = mu * inner(grad(u), grad(v))*dx + inner(b, grad(u) * v) * dx
	L = f * v * dx
	
	boundaries = FacetFunction('size_t', mesh)
	T_lb, T_rt = [T_LEFT_BOTTOM(), T_RIGHT_TOP()]
	T_lb.mark(boundaries, 1)
	T_rt.mark(boundaries, 2)

	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	
	
	infnorm = 1. / sqrt(2.)
	Pe = infnorm / (2. * N * mu)
	print Pe

	if SUPG:
		delta = 2. / (N * infnorm) * (coth(Pe) - 1. / Pe)
		S = delta * inner(b, grad(v)) * inner(b, grad(u)) * dx
		a += S
		L += delta * f * inner(b, grad(v)) * dx

	bcs = [DirichletBC(V, 0, boundaries, 1)]
	if not Neum:
		bcs.append(DirichletBC(V, 0, boundaries, 2))
		
	u_sol = Function(V)
	solve(a == L, u_sol, bcs)
    
	print min(u_sol.vector().array()) 
	
	if (show_plot):
		plot(u_sol)
		plt.title('sdaas')
		plt.show()
        File('sol.pvd') << u_sol

#adv_dif_equation(1e-3, 32, False, True)
adv_dif_equation(1e-3, 32, True, True, True)

