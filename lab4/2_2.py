from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


def Stokessten(stenosis = 0, Q = 10.):
	''' stenosis: 	0 = no stenosis
					1 = mild stenosis
					2 = severe stenosis
	'''

	# Create mesh
	if stenosis == 0:
		mesh = Mesh('no_stenosis/no_stenosis.xml');
		boundaries = MeshFunction('size_t', mesh, 'no_stenosis/no_stenosis_facet_region.xml')
	elif stenosis == 1:
		mesh = Mesh('stenosis_f0.4/stenosis_f0.4.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.4/stenosis_f0.4_facet_region.xml')
	else:
		mesh = Mesh('stenosis_f0.6/stenosis_f0.6.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6/stenosis_f0.6_facet_region.xml')
	
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	
	# Define function spaces
	VE = VectorElement('P', mesh.ufl_cell(), 2)
	PE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * PE
	W = FunctionSpace(mesh, ME)

	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)

	# Define boundary conditions
	''' boundaries: 1 = inflow
					2 = outflow
					3 = symmetric
					4 = no-slip
	'''		
			
	# Dirichlet inflow bc
	UQ = Q*3/4
	class Inflow(Expression):
		def __init__(self, Q, **kwargs):
			self.Q = Q

		def eval(self, values, x):
			values[0] = self.Q * (1 - x[1] * x[1])
			values[1] = 0

		def value_shape(self):
			return (2,)

	bcs = []
	expr = Inflow(UQ, degree = 2)
	bc = DirichletBC(W.sub(0), expr, boundaries, 1)
	bcs.append(bc)

	# Dirichlet part symmetric bc
	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 3)
	bcs.append(bc)

	# Dirichlet no-slip bc
	bc = DirichletBC(W.sub(0), [0,0], boundaries, 4)
	bcs.append(bc)

	# Define variational problem
	mu = 0.035
	a = mu * inner(grad(u), grad(v)) * dx + div(u) * q * dx - div(v) * p * dx 
	L = inner(Constant([0,0]),v) * dx # blijkbaar werkt gewoon = 0 niet

	# Compute solution
	w = Function(W)
	solve(a == L, w, bcs)

	# Split the mixed solution
	(u, p) = w.split()
	
	# Compute Q
	n = Constant([1,0])
	Q = assemble(2*inner(u,n)*ds(1))
	
	print 'Q = ', Q
	
	# plot both u and p in one figure
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

	fig.savefig('sol2.png')

	# Save solution
	ufile_pvd = File("Stokes2_2u.pvd")
	ufile_pvd << u
	pfile_pvd = File("stokes2_2p.pvd")
	pfile_pvd << p
	
	# Compute pressure drop
	dp = assemble(p * ds(1) - p * ds(2))     # 1 / |T_1| = 1 / |T_2| = 1
	return dp
	
print Stokessten(stenosis = 2, Q = 25.)