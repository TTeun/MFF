from dolfin import *

class T_HORI(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1.0) or near(x[1], 0.0))

class T_VERT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0))

class T_TOP(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 1.0))

class T_BOTTOM(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0))

class T_LEFT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0))

class T_RIGHT(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1.0))

def set_neumann_robin_boundary(mesh):
    boundaries = FacetFunction('size_t', mesh)
    T_v, T_h = [T_VERT(), T_HORI()]
    T_v.mark(boundaries, 0)
    T_h.mark(boundaries, 1)

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)    
    
    return [boundaries, ds]

def set_mixed_boundary(mesh):
    boundaries = FacetFunction('size_t', mesh)
    T_left, T_right, T_hori = [T_LEFT(), T_RIGHT(), T_HORI()]
    T_left.mark(boundaries, 0)
    T_right.mark(boundaries, 1)
    T_hori.mark(boundaries, 2)    

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)    
    
    return [boundaries, ds]