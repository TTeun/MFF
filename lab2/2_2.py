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
    u_sol = Function(V)

    # Set up variational problem
    u0 = Expression("9 + 2*x[0]*x[0] + 3*x[1]*x[1]", degree=2)
    f = Expression("-10", degree=2) + u0
    a = (u * v) * dx + inner(grad(u), grad(v)) * dx
    L = f * v * dx

    if (bc_type == 'dirichlet'):
        # Set up boundary
        boundaries = FacetFunction('size_t', mesh)
        bc = DirichletBC(V, u0, boundaries, 0)

        # Solve the system
        solve(a == L, u_sol, bc)
    else:
        if (bc_type == 'neumann'):
            # Set up boundary
            [boundaries, ds] = set_neumann_or_robin_boundary(mesh)

            # Set up boundary functions
            u_x = Expression('4 * x[0]', degree=1)
            u_y = Expression('6 * x[1]', degree=1)

            # Add terms from neumann condition
            L += u_y * v * ds(1) + u_x * v * ds(0)
            # Solve the system
            solve(a == L, u_sol)

        if (bc_type == 'robin'):
            # Set up boundary
            [boundaries, ds] = set_neumann_or_robin_boundary(mesh)

            # Set up boundary functions
            u_x = Expression('4 * x[0]', degree=1) + u0
            u_y = Expression('6 * x[1]', degree=1) + u0

            # Add terms from robin condition
            a += (u * v) * ds
            L += u_y * v * ds(1) + u_x * v * ds(0)
            # Solve the system
            solve(a == L, u_sol)

        if (bc_type == 'mixed'):
            # Set up boundary
            [boundaries, ds] = set_mixed_boundary(mesh)
            bc = DirichletBC(V, u0, boundaries, 0)

            # Set up boundary functions
            u_right = Expression('4', degree=1)
            u_y = Expression('6 * x[1]', degree=1) + u0

            # Add terms from robin and neumann condition
            a += (u * v) * ds(2)
            L += u_right * v * ds(1) + u_y * v * ds(2)
            # Solve the system
            solve(a == L, u_sol, bc)

    # Plot solution
    plot(u_sol, interactive=True)
    plt.title(bc_type)
    plt.show()

    # Assess the solution by using the exact solution in the errornorm
    print errornorm(u0, u_sol)

    if (print_to_file):
        # Print to files readable by ParaView
        file = File(bc_type + '.pvd')
        file << u

diffreac(64, 'dirichlet')
diffreac(64, 'neumann')
diffreac(64, 'robin')
diffreac(64, 'mixed')
