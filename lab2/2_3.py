from dolfin import *
from numpy import *
import matplotlib.pyplot as plt
#from fenics import * # here we need Point
import mshr

# supress uncecessary output
set_log_level(ERROR)



class T_Dir(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], -1.0) or near(x[0], 1.0))
        

class T_Neu(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], -1.0) or near(x[1], 1.0))
        
class T_Rob(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and -0.5<x[0]<0.5 and -0.5<x[1]<0.5  #

def set_boundary(mesh):
    boundaries = FacetFunction('size_t', mesh)
    T_1, T_2, T_3 = [T_Dir(), T_Neu(), T_Rob()]
    T_1.mark(boundaries, 0)
    T_2.mark(boundaries, 1)
    T_3.mark(boundaries, 2)

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    return [boundaries, ds]




def diffreac(h, print_to_file=False):
    N = 32
    k = 401
    u_f = 280

    # Create mesh 
    domain = mshr.Rectangle(Point(-1, -1), Point(1, 1)) - \
        mshr.Circle(Point(0, 0), 0.25)
    mesh = mshr.generate_mesh(domain, N)

    # Set up function space
    V = FunctionSpace(mesh, "Lagrange", 2)
    
    # Set up functions
    u = TrialFunction(V)
    v = TestFunction(V)
    u_sol = Function(V)

    # Set up variational problem
    f = Expression("0", degree=2)
    a = -k * inner(grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Set up boundary
    [boundaries, ds] = set_boundary(mesh)
    bc = DirichletBC(V, 1000, boundaries, 0)

    # Set up boundary functions
    g_N = Expression('0', degree=1)
    g_R = h * u_f
   

    # Add terms from robin and neumann condition
    a += h * (u * v) * ds(2)
    L += g_N * v * ds(1) + g_R * v * ds(2)

	
    # Solve the system
    solve(a == L, u_sol, bc)
	
    # Plot solution
    plot(u_sol, interactive=True)
    plt.title('h = %d.pvd' % h)
    plt.show()

    # Assess the solution by using the exact solution in the errornorm
    #print errornorm(u0, u_sol)

    if (print_to_file):
        # Print to files readable by ParaView
        file = File('hvalue%d.pvd' % h)
        file << u_sol

diffreac(10,True)
diffreac(10000,True)
