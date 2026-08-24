from lappy import Domain, Polgyon
import numpy as np

class ParametricDomain:
    def __init__(self):
        pass

    def __call__(self, p):
        """Returns the Domain object corresponding to the parameter vector p"""
        pass

def polygon_vertices(p, perim=1):
    """Builds a N-gon corresponding to the parameter vector p with perimeter `perim`, where p contains the x and y
    coordinates of N-2 of the vertices. Thus p must have length 2N-4, that is len(p) >= 2 and len(p) must be even."""

    # number of vertices is len(p)/2 + 2
    N = len(p)//2 + 2

    # extract x and y coordinates
    x,y = p[:N-2], p[N-2:]

    # vertices stored in complex form (real part = x coord, imag part = y coord)
    vertices = np.zeros(N, dtype='complex')
    vertices[1:-1] = x + 1j*y

    # remaining perimeter
    C = perim - np.abs(vertices[2:]-vertices[1:-1]).sum()
    if C <= 0:
        raise ValueError("No such polygon exists...")

    # x coordinate of first vertex (assuming y coord of 0, and last vertex at origin)
    vertices[0] = 0.5*(x[0] + C + y[0]**2/(x[0]-C))
    return vertices

class VertexPolygon(ParametricDomain):
    """A class for polygons parameterized by 'vertex' data. An n-sided VertexPolygon with perimeter `perim` represents
    a family of planar polygons that have a vertex at the origin, an adjacent vertex along the postive x-axis, and with
    the other n-2 vertices 
    
    Parameters
    ----------
    n : int
        Number of sides/vertices of the family of polygons.
    perim : float (defaults to 1.0)
        Perimeter of the family of polygons
    """
    def __init__(self, n, perim=1.0):
        self.n = n
        self.perim = perim

    def __call__(self, p):
        if len(p)//2 + 2 != self.n:
            raise ValueError(f"p must have length {2*self.n-4} with n={self.n}")
        vertices = polygon_vertices(p, self.perim)
