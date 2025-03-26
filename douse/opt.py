import numpy as np
import scipy.linalg as la
from .param import polygon_vertices, polygon_perimeter
from lappy import PolygonEVP

def eig_obj(p,eigs_target,perim_tol=1e-15):
    # check number of eigenvalues, convert targets to normalized reciprocals
    K = len(eigs_target)
    targets = normalized_reciprocals(eigs_target)
    print("targets=",targets)
    
    # get vertices from parameter vector
    vertices = polygon_vertices(p)
    print("vertices=",vertices)
    N = len(vertices)

    # get perimeter and perimeter gradient
    perim, perim_grad = polygon_perimeter(vertices,jac=True)
    print("perim=",perim)
    if np.abs(perim-1) > perim_tol:
        raise ValueError('perimeter is not 1')
    implicit_grad = -np.concatenate((perim_grad[1:-1].real,perim_grad[1:-1].imag))/(perim_grad[0].real)

    # get eigenvalues and eigenderivatives w.r.t vertices
    evp = PolygonEVP(10*vertices)
    eigs = evp.solve_eigs_ordered(K)[0]
    print(eigs)
    eigs_jac = np.zeros((K,2*N))
    for i in range(K):
        dz = evp.eig_grad(eigs[i])
        eigs_jac[i,:N] = dz.real
        eigs_jac[i,N:] = dz.imag

    # get normalized reciprocals
    nus, nus_jac = normalized_reciprocals(eigs,jac=True)

    # evaluate loss function
    loss, loss_grad = l2_loss(nus,targets,jac=True)

    # compose derivatives with chain rule
    obj_grad = (loss_grad.reshape(1,-1)@nus_jac)@(np.outer(eigs_jac[:,0],implicit_grad) + np.delete(eigs_jac,[0,N,N+1,2*N-1],axis=1))

    return loss, obj_grad

def normalized_reciprocals(x,jac=False):
    if jac:
        col = (1/x[1:]).reshape(-1,1)
        diag = -x[0]/(x[1:]**2)
        jacobian = np.hstack((col,np.diag(diag)))
        return x[0]/x[1:], jacobian
    return x[0]/x[1:]

def l2_loss(x,y,jac=False):
    diff = x-y
    if jac:
        return la.norm(diff)**2, 2*diff
    return la.norm(diff)**2

def dirichlet_eigenvalues(*args):
    pass