from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def stokes(p2p1 = True, epsi = 1):
	'''
	method = 'p2p1' or 'p1p1stab'
	'''

	class T_LEFT(SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (near(x[0], 0.0))

	class T_RIGHT(SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (near(x[0], Len) )
			
	class T_TOP(SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (near(x[1], 1.0))

	class T_BOTTOM(SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (near(x[1], 0.0))

        
        
	Len = 6.
	mu = 0.001
	ny = 32
	nx = 3*ny
	ny = ny/2
	mesh = RectangleMesh(Point(0,0), Point(Len,1), nx, ny)
	
	if p2p1:
		VE = VectorElement('P', mesh.ufl_cell(), 2)
	else:
		VE = VectorElement('P', mesh.ufl_cell(), 1)
		
	PE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * PE

	W = FunctionSpace(mesh, ME)

	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)
	
	a = mu * inner(grad(u), grad(v))*dx + div(u)*q*dx - div(v)*p*dx 

	if not p2p1:
		h = 0.01 #CellDiameter(mesh)
		stab = epsi * (h * h / mu) * inner(grad(p),grad(q)) * dx
		a += stab
		
	L = 0.


	# Split up the boundary
	boundaries = FacetFunction('size_t', mesh)
	T_l, T_r, T_t, T_b = [T_LEFT(), T_RIGHT(), T_TOP(), T_BOTTOM()]
	T_l.mark(boundaries, 1)
	T_r.mark(boundaries, 2)
	T_t.mark(boundaries, 3)
	T_b.mark(boundaries, 4)
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	bcs = []
	bc = DirichletBC(W.sub(0), [0,0], boundaries, 3)
	bcs.append(bc)
	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 4)
	bcs.append(bc)

	n = Constant([1,0])
	L += inner(v, n) * ds(1)

	w = Function(W)

	solve(a == L, w, bcs)

	u, p = w.split()  

	File('sol.pvd') << u





	# Just a figure and one subplot
	fig = plt.figure()

	plt.subplot(2, 1, 1)
	c = plot(u)
	plt.title('results')
	plt.colorbar(c)
	plt.ylabel('y')


	plt.subplot(2, 1, 2)
	d = plot(p)
	plt.colorbar(d)
	plt.xlabel('x')
	plt.ylabel('y')

	plt.show()

	fig.savefig('sol1.png')
	
stokes(p2p1 = False, epsi = 1)