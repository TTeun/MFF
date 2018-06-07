from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

'''
	we have to compute c_back, find best c_tgt and implement both SUPG and PSPG
'''


def transient_Nstokes(finemesh = False, theta = 1., Re = 10., outputfile = 'u1', timestab = 0, Tenan = False, PSPG = False, SUPG = False):
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
	tconst = Constant([0,1])

	# Create mesh
	if finemesh:
		mesh = Mesh('stenosis_f0.6_fine/stenosis_f0.6_fine.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6_fine/stenosis_f0.6_fine_facet_region.xml')
	else:
		mesh = Mesh('stenosis_f0.6/stenosis_f0.6.xml');
		boundaries = MeshFunction('size_t', mesh, 'stenosis_f0.6/stenosis_f0.6_facet_region.xml')
	
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# define elements and functionspace
	if SUPG or PSPG:
		VE = VectorElement('P', mesh.ufl_cell(), 1)
	else:
		VE = VectorElement('P', mesh.ufl_cell(), 2)
	FE = FiniteElement('P', mesh.ufl_cell(), 1)
	ME = VE * FE

	W = FunctionSpace(mesh, ME)
	W0 = FunctionSpace(mesh, VE)

	# Define functions
	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)

	# Set up the problem
	d = Expression("d * t", d = dt, t = theta, degree=1)
	d_minus = Expression("d * (1. - t)", d = dt, t = theta, degree=1)
	a = rho * inner( u , v) * dx + d * mu * inner(grad(u), grad(v)) * dx - dt * p * div(v) * dx + d * q * div(u) * dx

	u0 = interpolate(Constant([0,0]), W0)

	a += d * rho * inner( grad(u) * u0, v) * dx

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
	
	if (timestab == 1) or (timestab == 2) or (timestab == 4):
		c_back = 1
	elif timestab == 3:
		c_tgt = 0.0025

	for t in np.arange(dt, Tf + dt, dt):
		u_in.t = t + (theta-1) * dt
		
		
		a = rho * inner( u , v) * dx + d * mu * inner(grad(u), grad(v)) * dx - dt * p * div(v) * dx + d * q * div(u) * dx
		a += d * rho * inner( grad(u) * u0, v) * dx
		
		L = rho * inner( u0 , v) * dx - d_minus * mu * inner(grad(u0), grad(v)) * dx - d_minus * q * div(u0) * dx
		L -= d_minus * rho * inner( grad(u0) * u0, v) * dx
		
		if timestab == 1:
			a += d * c_back * inner(u0,n) * inner(u,v) * ds(2)
			L -= d_minus * c_back * inner(u0,n) * inner(u0,v) * ds(2)
		elif timestab == 2:
			a += d * c_back * 0.5 * abs(inner(u0,n)-abs(inner(u0,n))) * inner(u, v)* ds(2)
			L -= d_minus * c_back * 0.5 * abs(inner(u0,n)-abs(inner(u0,n))) * inner(u0, v)* ds(2)
		elif timestab == 3:
			a += d * c_tgt * rho / 4. * abs(inner(u0,n)-abs(inner(u0,n))) * inner(grad(u)*tconst, grad(v)*tconst) * ds(2)
			L -= d_minus * c_tgt * rho / 4. * abs(inner(u0,n)-abs(inner(u0,n))) * inner(grad(u0)*tconst, grad(v)*tconst) * ds(2)
		if Tenan:
			a += d * rho * 0.5 * div(u0) * inner(u, v) * dx
			
		assemble(a, tensor=A)
		[bc.apply(A) for bc in bcs]
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

transient_Nstokes(Re = 7000, outputfile = 'utest', timestab = 2, Tenan = True, SUPG = True)