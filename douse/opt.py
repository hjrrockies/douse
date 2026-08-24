import numpy as np
import scipy.linalg as la
from .param import polygon_vertices, polygon_perimeter
from lappy import PolygonEVP

def eig_obj(p,eigs_target,perim_tol=1e-15,evp_kwargs={'order':20},mps_kwargs={},eig_list=None,log=False,verbose=False):
    from .evp import PolygonEVP
    # check number of eigenvalues, convert targets to normalized reciprocals
    K = len(eigs_target)
    targets = normalized_reciprocals(eigs_target)
    
    # get vertices from parameter vector
    vertices = polygon_vertices(p)
    N = len(vertices)
    if verbose: print(f"Evaluating at p={np.array_str(p,precision=3)}")

    # get perimeter and perimeter gradient
    perim, perim_grad = poly_perim(vertices,jac=True)
    if np.abs(perim-1) > perim_tol:
        raise ValueError('perimeter is too high')
    implicit_grad = -np.concatenate((perim_grad[1:-1].real,perim_grad[1:-1].imag))/(perim_grad[0].real)
    vertices_jac = np.zeros((2*N,len(p)))
    vertices_jac[0] = implicit_grad
    vertices_jac[1:N-1,:len(p)//2] = np.eye(len(p)//2)
    vertices_jac[N+1:-1,len(p)//2:] = np.eye(len(p)//2)

    # rescale vertices so the first eigenvalue is O(1)
    A,P = polygon_area(vertices),1
    weyl_1 = ((P+np.sqrt(P**2+16*np.pi*A))/(2*A))**2

    # get eigenvalues and eigenderivatives w.r.t vertices
    evp = PolygonEVP(np.sqrt(weyl_1)*vertices,**evp_kwargs)
    eigs = evp.solve_eigs_ordered(K+1,ppl=20,mps_kwargs=mps_kwargs)[0][:K]
    if eig_list is not None:
        eig_list.append(weyl_1*eigs)
    eigs_jac = np.zeros((K,2*N))
    for i in range(K):
        dz = evp.eig_grad(eigs[i])
        eigs_jac[i,:N] = dz.real
        eigs_jac[i,N:] = dz.imag

    # get normalized reciprocals
    nus, nus_jac = normalized_reciprocals(eigs,jac=True)

    # evaluate loss function
    loss, loss_grad = l2_loss(nus,targets,log,jac=True)

    # compose derivatives with chain rule
    # obj_grad = (loss_grad.reshape(1,-1)@nus_jac)@(np.outer(eigs_jac[:,0],implicit_grad) + np.delete(eigs_jac,[0,N,N+1,2*N-1],axis=1))
    grad = (((loss_grad.reshape(1,-1)@nus_jac)@eigs_jac)@vertices_jac)[0]

    return loss, grad

def normalized_reciprocals(x,jac=False):
    if jac:
        col = (1/x[1:]).reshape(-1,1)
        diag = -x[0]/(x[1:]**2)
        jacobian = np.hstack((col,np.diag(diag)))
        return x[0]/x[1:], jacobian
    return x[0]/x[1:]

def l2_loss(x,y,log=False,jac=False):
    diff = x-y
    out = la.norm(diff)**2
    if jac:
        grad = 2*diff
        if log: return np.log(out),grad/out
        else: return out, grad
    if log: return np.log(out)
    else: return out