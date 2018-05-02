from dolfin import *
from numpy import *
import matplotlib.pyplot as plt
from fenics import * # here we need Point
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
        return on_boundary and -0.5<x[0]<0.5 and -0.5<x[1]<0.5  

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
    k = 401.
    u_f = 280.

    # Create mesh 
    domain = mshr.Rectangle(Point(-1, -1), Point(1, 1)) - mshr.Circle(Point(0, 0), 0.25)
    mesh = mshr.generate_mesh(domain, N)

    # Set up function space
    V = FunctionSpace(mesh, "Lagrange", 2)
    
    # Set up functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Set up boundary
    [boundaries, ds] = set_boundary(mesh)
    bc = DirichletBC(V, 10000., boundaries, 0)

    # Set up boundary functions
    g_N = Expression('0', degree=1)
    g_R = h * u_f
   
    # Set up variational problem
    a = k * inner(grad(u), grad(v)) * dx + k * (u * v) * ds(2)
    L = g_R * k * v * ds(2)
    u = Function(V)

    # Solve the system
    solve(a == L, u, bc)
	
    # Plot solution
    plot(u, interactive=True)
    plt.title('h = %d' % h)
    plt.show()

    print assemble(u * dx)

    if (print_to_file):
        # Print to files readable by ParaView
        file = File('hvalue%d.pvd' % h)
        file << u

diffreac(10.,True)
diffreac(10000.,True)
