import numpy as np
import pymps as lp

def polygon_vertices(p,perim=1,jac=False):
    """Builds a perimeter-1 N-gon corresponding to the parameter vector p, where p contains the x and y
    coordinates of N-2 of the vertices. Thus p must have length 2N-4, that is len(p) >= 2 and len(p) must be even."""

    # number of vertices is len(p)/2 + 2
    N = len(p)//2 + 2

    # extract x and y coordinates
    x,y = p[:N-2],p[N-2:]

    # vertices stored in complex form (real part = x coord, imag part = y coord)
    vertices = np.zeros(N,dtype='complex')
    vertices[1:-1] = x + 1j*y

    # remaining perimeter
    C = perim - np.abs(vertices[2:]-vertices[1:-1]).sum()
    if C <= 0:
        raise ValueError("No such polygon exists...")

    # x coordinate of first vertex (assuming y coord of 0, and last vertex at origin)
    vertices[0] = 0.5*(x[0] + C + y[0]**2/(x[0]-C))

    # optionally compute the jacobian of the vertices with respect to the parameter vector
    # note: it is better in practice to use the implicit form which comes from polygon_perimeter
    if jac:
        l = lp.edge_lengths(vertices)
        alpha = y[0]/(x[0]-C)
        jacobian = np.zeros((2*N,len(p)))
        x,y = vertices.real, vertices.imag

        # partial derivatives of x_1 w.r.t. to p
        jacobian[0,0] = 0.5*((1-alpha**2) + (1+alpha**2)*(x[2]-x[1])/l[1])
        jacobian[0,N-2] = alpha + 0.5*(1+alpha**2)*(y[2]-y[1])/l[1]
        jacobian[0,1:N-3] = 0.5*(1+alpha**2)*((x[3:-1]-x[2:-2])/l[2:-2]-(x[2:-2]-x[1:-3])/l[1:-3])
        jacobian[0,N-1:-1] = 0.5*(1+alpha**2)*((y[3:-1]-y[2:-2])/l[2:-2]-(y[2:-2]-y[1:-3])/l[1:-3])
        jacobian[0,N-3] = -0.5*(1+alpha**2)*(x[-2]/np.abs(vertices[-2]) + (x[-2]-x[-3])/l[-2])
        jacobian[0,-1] = -0.5*(1+alpha**2)*(y[-2]/np.abs(vertices[-2]) + (y[-2]-y[-3])/l[-2])

        # partial derivatives of x_j w.r.t. themselves are identity
        jacobian[1:N-1,:N-2] = np.eye(N-2)
        # partial derivatives of y_j w.r.t. themselves are identity
        jacobian[N+1:-1,N-2:] = np.eye(N-2)
        
        return vertices, jacobian
    return vertices

def polygon_perimeter(vertices,jac=False):
    """computes the perimeter, and optionally the gradient of the perimeter, of a polygon
    with given vertices, written in complex form"""
    dv = vertices-np.roll(vertices,1)
    lengths = np.abs(dv)
    perim = lengths.sum()

    # gradient of perimeter with respect to vertices (ideal for use in implicit representation of
    # perimeter-constrained polygon parameterization derivatives)
    if jac:
        grad = dv/lengths
        grad = grad - np.roll(grad,-1)
        return perim, grad
    return perim

def perim_constraint(p,perim=1):
    """Represents the constraint on the polygon perimeter"""
    # number of vertices is len(p)/2 + 2
    N = len(p)//2 + 2

    # extract x and y coordinates
    x,y = p[:N-2],p[N-2:]

    # vertices stored in complex form (real part = x coord, imag part = y coord)
    vertices = np.zeros(N-1,dtype='complex')
    vertices[:-1] = x + 1j*y

    return perim-polygon_perimeter(vertices)

def perim_constraint_grad(p,perim=1):
    """Represents the constraint on the polygon perimeter"""
    # number of vertices is len(p)/2 + 2
    N = len(p)//2 + 2

    # extract x and y coordinates
    x,y = p[:N-2],p[N-2:]

    # vertices stored in complex form (real part = x coord, imag part = y coord)
    vertices = np.zeros(N-1,dtype='complex')
    vertices[:-1] = x + 1j*y
    _,grad = polygon_perimeter(vertices,jac=True)
    jac = np.empty((1,p.shape[0]))
    jac[:N-2],jac[N-2:] = grad[:N-1].real, grad[:N-1].imag
    return jac

def boltz(x,y,alpha=10,jac=False):
    expax = np.exp(alpha*x)
    expay = np.exp(alpha*y)
    if not jac:
        return (x*expax + y*expay)/(expax+expay)
    else:
        out = (x*expax + y*expay)/(expax+expay)
        grad = np.zeros(2)
        grad[0] = (expax/(expax+expay))*(1+alpha*(x-out))
        grad[1] = (expay/(expax+expay))*(1+alpha*(y-out))
        return out,grad
    
def d(a,b,c,jac=False):
    if not jac:
        return (b.real-a.real)*(c.imag-a.imag) - (b.imag-a.imag)*(c.real-a.real)
    else:
        out = (b.real-a.real)*(c.imag-a.imag) - (b.imag-a.imag)*(c.real-a.real)
        grad = np.zeros(3,dtype='complex')
        grad[0] = -(c.imag-a.imag)+(b.imag-a.imag)-1j*((b.real-a.real)+(c.real-a.real))
        grad[1] = (c.imag-a.imag) + 1j*(c.real-a.real)
        grad[3] = -(b.imag-a.imag)+1j*(b.real-a.real)
        return out,grad

def valid_polygon(vertices):
    N = len(vertices)
    out = np.zeros(N*(N-3)//2) # number of non-adjacent edges to compare against
    v = vertices
    k = 0
    for i in range(2,N-1):
        a = d(v[0],v[-1],v[i-1])*d(v[0],v[-1],v[i])
        b = d(v[i-1],v[i],v[-1])*d(v[i-1],v[i],v[0])
        out[k] = boltz(a,b)
        k += 1
    for i in range(1,N-2):
        for j in range(i+2,N):
            a = d(v[i-1],v[i],v[j-1])*d(v[i-1],v[i],v[j])
            b = d(v[j-1],v[j],v[i-1])*d(v[j-1],v[j],v[i])
            out[k] = boltz(a,b)
            k += 1
    return out


