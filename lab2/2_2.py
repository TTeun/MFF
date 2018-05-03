from dolfin import *
from numpy import *
import matplotlib.pyplot as plt
from domains import *

# supress uncecessary output
set_log_level(ERROR)


def diffreac(N,  bc_type='dirichlet', print_to_file=False):
    # Create mesh and define function space
    mesh = UnitSquareMesh(N, N)

    # Set up function space
    V = FunctionSpace(mesh, "Lagrange", 2)

    # Set up functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Set up variational problem
    u0 = Expression("9 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)
    f = Expression("-10", degree=0) + u0
    a = (u * v) * dx + inner(grad(u), grad(v)) * dx
    L = f * v * dx

    bcs = []

    if (bc_type == 'dirichlet'):
        # Set up boundary
        def bound(x, on_boundary):
            return on_boundary
            
        bc = DirichletBC(V, u0, bound)
        bcs.append(bc)

    elif (bc_type == 'neumann'):
        # Set up boundary
        [boundaries, ds] = set_neumann_or_robin_boundary(mesh)

        # Set up boundary functions
        u_x = Expression('4 * x[0]', degree=1)
        u_y = Expression('6 * x[1]', degree=1)

        # Add terms from neumann condition
        L += u_y * v * ds(2) + u_x * v * ds(1)

    elif (bc_type == 'robin'):
        # Set up boundary
        [boundaries, ds] = set_neumann_or_robin_boundary(mesh)

        # Set up boundary functions
        u_x = Expression('4 * x[0]', degree=1) + u0
        u_y = Expression('6 * x[1]', degree=1) + u0

        # Add terms from robin condition
        a += (u * v) * ds
        L += u_y * v * ds(2) + u_x * v * ds(1)

    elif (bc_type == 'mixed'):
        # Set up boundary
        [boundaries, ds] = set_mixed_boundary(mesh)
        bc = DirichletBC(V, u0, boundaries, 1)
        bcs.append(bc)

        # Set up boundary functions
        u_right = Expression('4', degree=1)
        u_y = Expression('6 * x[1]', degree=1) + u0

        # Add terms from robin and neumann condition
        a += (u * v) * ds(3)
        L += u_right * v * ds(2) + u_y * v * ds(3)


    # Solve the system
    u_sol = Function(V)
    solve(a == L, u_sol, bcs)

    print errornorm(u0, u_sol)

    # Plot solution
    plot(u_sol, interactive=True)
    plt.title(bc_type)
    plt.show()

    if (print_to_file):
        # Print to files readable by ParaView
        file = File(bc_type + '.pvd')
        file << u_sol

diffreac(64, 'dirichlet')
diffreac(64, 'neumann')
diffreac(64, 'robin')
diffreac(64, 'mixed')
