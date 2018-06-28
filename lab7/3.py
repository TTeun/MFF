from dolfin import *
import matplotlib.pyplot as plt
import numpy as np



def CT_transient_Nstokes(
	Re = 10., 
	outputfile = 'u1',
	dt = 0.001):
	'''
	timestab =  0: no stabilization
				1-3: stabilization methods 1-3 of excersize 2.2
	'''
	# Some constants
	mu = 0.035
	rho = 1.2
	Tf = 0.4
	R = 1
	u_bulk = Re * mu / (2. * rho * R)
	n = Constant([1,0])

	u_in = Expression(('3/2 * u_bulk * sin(pi * t / Tf) * (1 - (x[1] / R) * (x[1] / R))', '0'), u_bulk = u_bulk, t = 0., Tf = Tf, R = R, degree=3)

	n = Constant([1,0])

	# Create mesh
	mesh = Mesh('channel/channel.xml');
	boundaries = MeshFunction('size_t', mesh, 'channel/channel_facet_region.xml')
	
	ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

	# define elements and functionspace
	
	V = VectorFunctionSpace(mesh, 'P', 1)
	Q = FunctionSpace(mesh, 'P', 1)
	
	# Define functions
	v = TestFunction(V)
	q = TestFunction(Q)
	u = TrialFunction(V)
	p = TrialFunction(Q)


	
	u0 = interpolate(Constant([0,0]), V)
	
	# Set up the problem
	
	# Viscous
	a_vis = rho * inner( u , v) * dx + dt * mu * inner(grad(u), grad(v)) * dx
	a_vis += dt * rho * inner( grad(u) * u0, v) * dx
	a_vis += dt * rho * 0.5 * div(u0) * inner(u, v) * dx 
	
	bcs_vis = []
	bc = DirichletBC(V, u_in, boundaries, 1)
	bcs_vis.append(bc)
	
	bc = DirichletBC(V.sub(1), 0, boundaries, 3)
	bcs_vis.append(bc)

	bc = DirichletBC(V, [0,0], boundaries, 4)
	bcs_vis.append(bc)

	A_vis = assemble(a_vis)
	[bc.apply(A_vis) for bc in bcs_vis]
	
	solver_vis = LinearSolver()
	solver_vis.set_operator(A_vis)
	
	# Pressure projection
	a_pr = inner(grad(p), grad(q)) * dx
	
	bc_pr = DirichletBC(Q, 0, boundaries, 2)
	
	A_pr = assemble(a_pr)
	bc_pr.apply(A_pr)
	
	solver_pr = LinearSolver()
	solver_pr.set_operator(A_pr)
	
	
	# Velocity correction
	a_vel = inner(u, v) * dx
	
	A_vel = assemble(a_vel)
	
	solver_vel = LinearSolver()
	solver_vel.set_operator(A_vel)
	
	# backflow stabilization
	c_back = rho / 2
	a_vis += dt * c_back * 0.5 * abs(inner(u0,n)-abs(inner(u0,n))) * inner(u, v)* ds(2)
	

	solu = XDMFFile(outputfile + '_u.xdmf')
	solu.parameters['rewrite_function_mesh'] = False
	
	solp = XDMFFile(outputfile + '_p.xdmf')
	solp.parameters['rewrite_function_mesh'] = False
	
	u_tilde = Function(V)
	pnew = Function(Q, name = 'pressure')
	u1 = Function(V, name = 'solution')
	i = 0;
	mass_bal = []
	tt = []
	for t in np.arange(dt, Tf + dt, dt):
		u_in.t = t
		
		# Viscous Step
		
		assemble(a_vis, tensor=A_vis)
		[bc.apply(A_vis) for bc in bcs_vis]
		L_vis = rho * inner( u0 , v) * dx
		b = assemble(L_vis)
		[bc.apply(b) for bc in bcs_vis]
		solver_vis.solve(u_tilde.vector(), b)	
		
		
		# Pressure projection Step
		
		L_pr = - rho / dt * div(u_tilde) * q * dx
		b = assemble(L_pr)
		bc_pr.apply(b)
		solver_pr.solve(pnew.vector(), b)
		
		# Velocity correction Step
		
		L_vel = inner(u_tilde,v) * dx  - dt / rho * inner(grad(pnew), v)  * dx
		b = assemble(L_vel)
		solver_vel.solve(u1.vector(), b)
		u0 = u1
		
		print t
		if ( t % 0.01 == 0):
			solu.write(u1, t)
			solp.write(pnew, t)
		
		mass_bal.append(assemble(-inner(u1,n)*ds(1) + inner(u1,n)*ds(2)))
		tt.append(t)
		i += 1
		
	
	fig = plt.figure()
	plt.plot(tt, mass_bal)
	plt.title('mass balance for dt = ' + str(dt))
	plt.xlabel('t')
	plt.ylabel('mass balance')
	fig.savefig('mass_bal_' + outputfile + '.png')
	
		
		

CT_transient_Nstokes(Re = 5000, outputfile = '1_3_dt1e-3', dt = 0.01)
