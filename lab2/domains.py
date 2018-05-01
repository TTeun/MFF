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
