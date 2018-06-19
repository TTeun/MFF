from dolfin import *
import matplotlib.pyplot as plt
import numpy as np



def monolithic_transient_stokes(
	Re = 10., 
	outputfile = 'u1',
	Tct = 0.001):
	'''
	timestab =  0: no stabilization
				1-3: stabilization methods 1-3 of excersize 2.2
	'''
	# Some constants
	mu = 0.035
	rho = 1.2
	Tf = 0.4
	dt = 0.01
	R = 1
	u_bulk = Re * mu / (2. * rho * R)

	u_in = Expression(('3/2 * u_bulk * sin(pi * t / Tf) * (1 - (x[1] / R) * (x[1] / R))', '0'), u_bulk = u_bulk, t = 0., Tf = Tf, R = R, degree=3)

	n = Constant([1,0])

	# Create mesh
	mesh = Mesh('stenosis_f0.6_fine/stenosis_f0.6_fine.xml');
	boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6_fine/stenosis_f0.6_fine_facet_region.xml')
	
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# define elements and functionspace
	VE = VectorElement('P', mesh.ufl_cell(), 1)
	FE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * FE

	W = FunctionSpace(mesh, ME)
	W0 = FunctionSpace(mesh, VE)

	# Define functions
	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)

	pspg_term = Tct/rho
	
	# Set up the problem
	a = rho * inner( u , v) * dx + dt * mu * inner(grad(u), grad(v)) * dx - dt * p * div(v) * dx + dt * q * div(u) * dx
	a += pspg_term * inner(grad(p), grad(q)) * dx # PSPG
	
	
	u0 = interpolate(Constant([0,0]), W0)


	# Add Dirichlet bc's
	bcs = []
	bc = DirichletBC(W.sub(0), u_in, boundaries, 1)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0).sub(1), 0, boundaries, 3)
	bcs.append(bc)

	bc = DirichletBC(W.sub(0), [0,0], boundaries, 4)
	bcs.append(bc)

	A = assemble(a)
	[bc.apply(A) for bc in bcs]

	sol = XDMFFile(outputfile + '.xdmf')
	sol.parameters['rewrite_function_mesh'] = False
	

	pspg_term = Tct/rho

	for t in np.arange(dt, Tf + dt, dt):
		u_in.t = t
		

		
		L = rho * inner( u0 , v) * dx
		

		

			
		b = assemble(L)
		[bc.apply(b) for bc in bcs]
		w = Function(W, name='solution')
		solve(A, w.vector(), b)
		u0, _ = split(w)
		
		print t
		sol.write(w.split()[0], t)
		
        
#transient_Nstokes(Re = 10, outputfile = 'u1_Re10')
#transient_Nstokes(Re = 500, outputfile = 'u1_Re500')
#transient_Nstokes(Re = 2000, outputfile = 'u1_Re2000')

monolithic_transient_stokes(Re = 7000, outputfile = 'utest', Tct = 0.001)