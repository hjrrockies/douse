"""Maximizing `lam_k/lam_1` through a degeneracy.

WHY THE SMOOTH METHOD IS NOT ENOUGH. Measured on the runs in `benchmarks/ratio_stall.py`:
backtracking ascent stalls 5.7e-04 short of the regular 4-gon and 1.15e-02 short of the regular
5-gon, and the shape it stops at is not regular -- the pentagon run ends with sides spanning
0.716 to 1.000 and angles 94.8 to 126.5 against 108. The diagnostic is unmistakable: `|grad|`
never falls, sitting at 1.3-2.1 throughout and largest at the LAST iterate. A smooth optimum has
a vanishing gradient; that one is grinding against a corner it cannot turn.

The corner is a degeneracy. At every regular N-gon `lam_2 = lam_3` exactly, so `lam_2` is only
directionally differentiable there and no gradient exists to vanish.

WHAT REPLACES IT. At a cluster of size m the modes emerging under a perturbation `d` are the
eigenvalues of `M(d) = sum_i d_i M^(i)`, LINEAR in `d`. So the local model of the objective is

    dF(d) = [ sigma_j(M(d)) * lam_1  -  lam_k * (g1.d) ] / lam_1^2

with `sigma_j` the j-th ordered eigenvalue. When `lam_k` is the LOWEST member of its cluster --
measured to be the case for k = 2 at every regular N-gon, and for k = 4 when N >= 5 --
`sigma_0 = lam_min` is CONCAVE in `d`, so maximizing the model over a trust region is a concave
program with one optimum. That is the good case, and it is the one this module handles.

    maximize   t * lam_1 - lam_k * (g1.d)
    subject to Lambda + M(d) >= t I,   ||d|| <= Delta

an SDP in general. For m = 2 the matrix inequality is just two scalar conditions --
`tr(Lambda + M(d) - tI) >= 0` and `det(Lambda + M(d) - tI) >= 0` -- so SLSQP solves it and no
SDP solver is needed. That matters: D_N gives m = 2 generically, so the cheap case is also the
common one.

`Lambda = diag(lam_lo - lam_k, ..., lam_hi-1 - lam_k)` IS NOT OPTIONAL, and leaving it out was
the first version's mistake. The cluster is eps-ACTIVE, so its members are near-equal rather
than equal -- at the points these runs stop, split by around 1e-03. Modelling them as exactly
degenerate makes the splitting look free and inflates the predicted gain, so the trust region
keeps rejecting steps and collapses. Measured before the fix: N=5 collapsed to a radius of
5.7e-15 while still 1.9e-03 short, and N=4 ran out of iterations with the sides equal to 7.7e-05
but the angles still 0.124 degrees off -- a rhombus, not a square. With `Lambda` the model is
exact at an exact degeneracy and correct in the approach to one.

THE STOPPING TEST IS THE CERTIFICATE. The subproblem's optimal value is the best first-order
improvement available; when it is zero, no direction improves `F` to first order and `p` is
first-order optimal. Unlike the determinant condition in `douse.certify`, this does not hold
automatically -- at a non-optimal point the model returns a strictly positive gain and a
direction to move in.

WHAT IS NOT HANDLED. `lam_k` the HIGHEST member of its cluster (k = 3 always, k = 5 for N >= 5)
gives `sigma_{m-1} = lam_max`, which is CONVEX; maximizing it is a genuinely harder problem with
no unique local model optimum. `_refuse_convex_case` says so rather than returning a number.
"""
import warnings

import numpy as np
from scipy.optimize import minimize

from .derivative import cluster_matrices, dlams_dp
from .optimize import OptimizeResult

__all__ = ['local_model', 'trust_region_step', 'maximize_ratio']


def _active_cluster(lams, k, eps_rel):
    """Indices `[lo, hi)` of eigenvalues within `eps_rel` (relative) of `lam_k`.

    An EPS-ACTIVE set rather than an exact-degeneracy test. The gap closes continuously as the
    optimizer approaches, so a rule that only fires at an exact tie would switch models one step
    before the end and never get the benefit; and one that fires too eagerly spends solves on
    cluster matrices it does not need. `eps_rel = 1e-3` puts the switch around where the measured
    stall begins.
    """
    lams = np.asarray(lams, dtype=float)
    lo = hi = k
    while lo > 0 and (lams[k] - lams[lo - 1]) <= eps_rel*abs(lams[k]):
        lo -= 1
    while hi + 1 < len(lams) and (lams[hi + 1] - lams[k]) <= eps_rel*abs(lams[k]):
        hi += 1
    return lo, hi + 1


def local_model(family, p, solver, lams, k=1, eps_rel=1e-3, align_rtol=1e-9):
    """The first-order model of `lam_k/lam_1` at `p`.

    Returns a dict with `mode` ('smooth' or 'cluster'), the pieces the trust-region step needs,
    and the cluster range so a caller can see the model switch happening.
    """
    lams = np.asarray(lams, dtype=float)
    lo, hi = _active_cluster(lams, k, eps_rel)
    m, j = hi - lo, k - lo
    lam_1, lam_k = float(lams[0]), float(lams[k])
    out = dict(cluster=(lo, hi), m=m, j=j, lam_1=lam_1, lam_k=lam_k, ratio=lam_k/lam_1)

    if m == 1:
        J = dlams_dp(family, p, solver, [lam_1, lam_k]) if k != 0 else \
            dlams_dp(family, p, solver, [lam_1])
        g1, gk = J[0], J[-1]
        out.update(mode='smooth', g1=g1, gk=gk,
                   grad=(gk*lam_1 - lam_k*g1)/lam_1**2)
        return out

    g1 = dlams_dp(family, p, solver, [lam_1])[0]
    split = float(lams[hi - 1] - lams[lo])
    aligned = split > align_rtol*abs(lam_k)
    # Aligned basis when the split is resolvable, so that `Lam` and `Ms` live in the SAME basis;
    # the cluster basis with `Lam = 0` when it is not, where any orthonormal basis serves.
    Ms = cluster_matrices(family, p, solver, float(np.mean(lams[lo:hi])), m,
                          lams=(lams[lo:hi] if aligned else None))
    Lam = np.diag(lams[lo:hi] - lam_k) if aligned else np.zeros((m, m))
    out.update(mode='cluster', g1=g1, Ms=Ms, Lam=Lam, split=split, aligned=aligned)
    return out


def _refuse_convex_case(model, k):
    if model['mode'] == 'cluster' and model['j'] != 0:
        raise NotImplementedError(
            f'lam_{k+1} is member {model["j"]} of a {model["m"]}-fold cluster, i.e. not its '
            f'lowest. Its directional derivative is then a HIGHER ordered eigenvalue of M(d), '
            f'which is convex rather than concave in d, and maximizing a convex function has no '
            f'unique local model optimum -- a trust-region step would pick an arbitrary vertex. '
            f'This is the k = 3 case (and k = 5 for N >= 5); it needs a different method, not a '
            f'looser tolerance.')


def trust_region_step(model, delta):
    """`(d, predicted_gain)` maximizing the local model over `||d|| <= delta`.

    Smooth case: steepest ascent to the trust boundary. Cluster case: the concave program in the
    module docstring, solved as an NLP in `(d, t)` -- for m = 2 the matrix inequality is two
    scalar constraints, so this needs no SDP solver.
    """
    if model['mode'] == 'smooth':
        g = model['grad']
        n = float(np.linalg.norm(g))
        if n == 0:
            return np.zeros_like(g), 0.0
        return delta*g/n, delta*n

    Ms, g1 = model['Ms'], model['g1']
    lam_1, lam_k = model['lam_1'], model['lam_k']
    n_p = len(g1)
    if Ms.shape[1] != 2:
        raise NotImplementedError(f'cluster of size {Ms.shape[1]}: the explicit 2x2 constraints '
                                  f'do not apply. An SDP solver (cvxpy) would be needed here.')
    A, B, C = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    Lam = model['Lam']
    l0, l1 = float(Lam[0, 0]), float(Lam[1, 1])       # offsets of the cluster members

    def unpack(z):
        return z[:n_p], z[n_p]

    def neg_gain(z):
        d, t = unpack(z)
        return -(t*lam_1 - lam_k*(g1 @ d))/lam_1**2

    def neg_gain_grad(z):
        d, t = unpack(z)
        out = np.zeros(n_p + 1)
        out[:n_p] = lam_k*g1/lam_1**2
        out[n_p] = -lam_1/lam_1**2
        return out

    cons = [
        # tr(Lambda + M(d) - t I) >= 0
        dict(type='ineq',
             fun=lambda z: (A @ z[:n_p] + l0) + (C @ z[:n_p] + l1) - 2*z[n_p]),
        # det(Lambda + M(d) - t I) >= 0
        dict(type='ineq',
             fun=lambda z: ((A @ z[:n_p] + l0) - z[n_p])*((C @ z[:n_p] + l1) - z[n_p])
             - (B @ z[:n_p])**2),
        # trust region
        dict(type='ineq', fun=lambda z: delta**2 - z[:n_p] @ z[:n_p]),
    ]
    z0 = np.zeros(n_p + 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(neg_gain, z0, jac=neg_gain_grad, constraints=cons, method='SLSQP',
                       options=dict(maxiter=200, ftol=1e-14))
    d, t = unpack(res.x)
    gain = -neg_gain(res.x)
    if not np.isfinite(gain) or gain < 0:
        return np.zeros(n_p), 0.0
    return d, float(gain)


def maximize_ratio(evaluator, p0, k=1, delta0=0.05, delta_max=0.5, eps_rel=1e-3,
                   max_iter=200, gain_tol=1e-7, eta=0.1, callback=None):
    """Trust-region ascent on `lam_k/lam_1`, valid through a degeneracy.

    `evaluator` must be a `SpectrumEvaluator` with `degenerate_tol=0` -- the guard that normally
    refuses a merged spectrum has to be off, because handling it is the point here.

    Stops when the model's best first-order gain falls below `gain_tol`, which is the
    first-order optimality certificate, not a step-size heuristic.
    """
    if evaluator.degenerate_tol > 0:
        raise ValueError('maximize_ratio handles degeneracies itself; construct the evaluator '
                         'with degenerate_tol=0 so it does not refuse them first')
    fam = evaluator.family
    p = np.array(p0, dtype=float)
    delta, message, success = float(delta0), f'reached max_iter={max_iter}', False
    n_cluster_steps, it = 0, 0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # (p, lams, F, solver) is ONE consistent state, replaced together or not at all. An
        # earlier version re-evaluated `lams` at the top of each iteration without recomputing
        # `F`, so the two drifted -- and the re-evaluation continued from whatever the last
        # REJECTED trial point had left in the tracker. It reported a ratio of 2.5400 for a
        # pentagon whose true ratio was 2.5124, above the Payne--Polya--Weinberger bound of
        # 2.5387 that no planar domain can exceed. The stored eigenvalues were right the whole
        # time; only the reported objective was stale.
        lams, solver = evaluator.eigenvalues(p)
        F = float(lams[k]/lams[0])

        for it in range(1, max_iter + 1):
            model = local_model(fam, p, solver, lams, k=k, eps_rel=eps_rel)
            _refuse_convex_case(model, k)
            if model['mode'] == 'cluster':
                n_cluster_steps += 1

            d, gain = trust_region_step(model, delta)
            # Criticality is gain PER UNIT RADIUS. The raw gain scales with `delta`, so testing
            # it directly declares optimality whenever the radius has shrunk -- measured: a run
            # stopping at gain 6.8e-05 while gain/delta was still 1.4, i.e. an O(1) directional
            # derivative and nowhere near optimal.
            crit = gain/delta if delta > 0 else 0.0
            if crit <= gain_tol:
                success = True
                message = (f'criticality {crit:.3g} (gain {gain:.3g} at radius {delta:.3g}) '
                           f'below gain_tol: first-order optimal')
                break

            q = p + d
            if not fam.is_valid(q):
                delta *= 0.5
                continue
            lams_q, solver_q = evaluator.eigenvalues(q)
            F_q = float(lams_q[k]/lams_q[0])
            rho = (F_q - F)/gain if gain > 0 else -np.inf

            if rho > eta:
                p, F, lams, solver = q, F_q, lams_q, solver_q
                if rho > 0.75:
                    delta = min(2*delta, delta_max)
            else:
                delta *= 0.5
                # The tracker's continuation state now points at the rejected trial; reseed it
                # from the accepted iterate so the next evaluation continues from where we are.
                evaluator._last = (p.copy(), np.asarray(lams).copy())
            if callback is not None:
                callback(p, F, model, delta, rho)
            if delta < 1e-14:
                message = f'trust radius collapsed to {delta:.3g}'
                break

    gap = (float(np.min(np.diff(np.sort(lams))/np.sort(lams)[:-1]))
           if len(lams) > 1 else np.inf)
    return OptimizeResult(x=p, fun=F, nit=it, lams=lams, gap=gap, delta=delta,
                          n_cluster_steps=n_cluster_steps, success=success, message=message)
