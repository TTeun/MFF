from dolfin import *
import matplotlib.pyplot as plt
import numpy as np




class T_LEFT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0))

class T_RIGHT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], Len) )
		
class T_TOP_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], -1.0) or near(x[1], 1.0))

Len = 6.
mu = 0.001
ny = 32
nx = 3*ny
mesh = RectangleMesh(Point(0,-1), Point(Len,1), nx, ny)

VE = VectorElement('P', mesh.ufl_cell(), 2)
PE = FiniteElement('P', mesh.ufl_cell(), 1)
ME = VE * PE

W = FunctionSpace(mesh, ME)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = mu * inner(grad(u), grad(v))*dx + div(u)*q*dx - div(v)*p*dx 

L = 0


# Split up the boundary
boundaries = FacetFunction('size_t', mesh)
T_l, T_r, T_tb = [T_LEFT(), T_RIGHT(), T_TOP_BOTTOM()]
T_l.mark(boundaries, 1)
T_r.mark(boundaries, 2)
T_tb.mark(boundaries, 3)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

bc = DirichletBC(W.sub(0), [0,0], boundaries, 3)

n = Constant([1,0])
L += inner(v, n) * ds(1)

w = Function(W)

solve(a == L, w, bc)

u, p = w.split()  

File('sol.pvd') << u




'''
c = plot(u)
plt.title('Flow')
plt.colorbar(c)
plt.show()
'''


# Just a figure and one subplot
plt.subplot(2, 1, 1)
c = plot(u)
plt.title('A tale of 2 subplots')
plt.colorbar(c)
plt.ylabel('u')


plt.subplot(2, 1, 2)
d = plot(p)
plt.colorbar(d)
plt.xlabel('time (s)')
plt.ylabel('p')

plt.show()