from dolfin import *
import numpy as np



def CT_transient_Nstokes(
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
	
	

	solu = XDMFFile(outputfile + '_u.xdmf')
	solu.parameters['rewrite_function_mesh'] = False
	
	solp = XDMFFile(outputfile + '_p.xdmf')
	solp.parameters['rewrite_function_mesh'] = False
	
	u_tilde = Function(V)
	pnew = Function(Q, name = 'pressure')
	u1 = Function(V, name = 'solution')
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
		solu.write(u1, t)
		solp.write(pnew, t)
		

def CT_transient_Nstokes_slow(
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
	
	

	solu = XDMFFile(outputfile + '_u.xdmf')
	solu.parameters['rewrite_function_mesh'] = False
	
	solp = XDMFFile(outputfile + '_p.xdmf')
	solp.parameters['rewrite_function_mesh'] = False
	
	u_tilde = Function(V)
	pnew = Function(Q, name = 'pressure')
	u1 = Function(V, name = 'solution')
	for t in np.arange(dt, Tf + dt, dt):
		u_in.t = t
		
		# Viscous Step
		a_vis = rho * inner( u , v) * dx + dt * mu * inner(grad(u), grad(v)) * dx
		a_vis += dt * rho * inner( grad(u) * u0, v) * dx
		a_vis += dt * rho * 0.5 * div(u0) * inner(u, v) * dx 
		L_vis = rho * inner( u0 , v) * dx
		
		bcs_vis = []
		bc = DirichletBC(V, u_in, boundaries, 1)
		bcs_vis.append(bc)
		
		bc = DirichletBC(V.sub(1), 0, boundaries, 3)
		bcs_vis.append(bc)

		bc = DirichletBC(V, [0,0], boundaries, 4)
		bcs_vis.append(bc)
		
		solve(a_vis == L_vis, u_tilde, bcs_vis)	
		
		# Pressure projection Step
		a_pr = inner(grad(p), grad(q)) * dx
		L_pr = - rho / dt * div(u_tilde) * q * dx
		
		bc_pr = DirichletBC(Q, 0, boundaries, 2)
		
		solve(a_pr == L_pr, pnew, bc_pr)
		
		# Velocity correction Step
		a_vel = inner(u, v) * dx
		L_vel = inner(u_tilde,v) * dx  - dt / rho * inner(grad(pnew), v)  * dx
		
		solve(a_vel == L_vel, u1)
		u0 = u1
		
		print t
		solu.write(u1, t)
		solp.write(pnew, t)
		
timer = Timer()
CT_transient_Nstokes(Re = 5000, outputfile = 'test', Tct = 0.001)
time = timer.elapsed()

timer = Timer()
CT_transient_Nstokes_slow(Re = 5000, outputfile = 'test', Tct = 0.001)
time_slow = timer.elapsed()

print('time slow method: ' )
print(time_slow)
print('time optimized method: ')
print(time)