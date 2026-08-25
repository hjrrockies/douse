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
eigenvalues of `Lambda + M(d)` with `M(d) = sum_i d_i M^(i)`, LINEAR in `d`. So the local model
of the objective is

    dF(d) = [ sigma_j(Lambda + M(d)) * lam_1  -  lam_k * (g1.d) ] / lam_1^2

with `sigma_j` the j-th ordered eigenvalue and `j` the position of `lam_k` inside its cluster.
`model_gain` evaluates exactly that, for any `d`, and it is what the trust-region ratio is
measured against.

ONE REDUCTION COVERS BOTH EXTREME POSITIONS. For a symmetric matrix,

    lam_max(Lambda + M(d)) = max_{||u||=1} [ u^T Lambda u + m(u).d ],   m(u)_i = u^T M^(i) u

so when `lam_k` is the HIGHEST member of its cluster (j = m-1) the subproblem is, exactly,

    max_{||d||<=Delta} dF(d) = max_{||u||=1} [ u^T Lambda u * lam_1 + Delta*||m(u) lam_1 - lam_k g1|| ] / lam_1^2

a maximization over the unit sphere `S^{m-1}` -- one-dimensional for m = 2, and a small
multi-start problem for m >= 3. That is `_step_convex`. The whole "convex case is a research
obstacle" reading was wrong: exchanging the two maxima removes it. See
`docs/spectral_optimization.tex` S6.2 for the m = 2 derivation this generalizes.

When `lam_k` is the LOWEST member (j = 0) the same object appears with the maxima NOT
exchangeable -- `lam_min` is concave, so it is a genuine max-min and the subproblem

    maximize   t * lam_1 - lam_k * (g1.d)
    subject to Lambda + M(d) >= t I,   ||d|| <= Delta

is a semidefinite program. For m = 2 the matrix inequality is two scalar conditions --
`tr(Lambda + M(d) - tI) >= 0` and `det(Lambda + M(d) - tI) >= 0` -- so SLSQP solves it and no SDP
solver is needed. That matters: D_N gives m = 2 generically, so the cheap case is also the
common one. For m >= 3 it goes to `cvxpy`, imported lazily so the common case never needs it.

INTERIOR POSITIONS (0 < j < m-1) are neither convex nor concave, and admit no reduction. They
do not need one: `model_gain` is an `m x m` `eigvalsh`, so the exact model can simply be
maximized directly by multi-start SLSQP -- see `_step_interior`, which measures within 1.3e-05
of a 200-start brute force. It is the one branch without a certificate, and
`OptimizeResult.n_interior_steps` counts how often a run used it.

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

HOW WIDE IS THE CLUSTER. `eps_rel` used to be pinned at 1e-3, which is a guess about where a
stall begins rather than a statement about the iterate. The principled width is "whatever a step
of radius `Delta` could merge": `M(d)` moves the cluster members by at most
`||d|| * ||(||M^(i)||)_i||`, so the eps-active set should be that wide and no wider. With
`eps_rel='auto'` (the default) `maximize_ratio` supplies `eps_abs = Delta * scale` each iterate,
`scale` being that norm from the previous model. It shrinks with the trust radius, which is what
was wanted: a converged run ends with an active set containing exactly the modes that are
genuinely degenerate.
"""
import warnings

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from . import degeneracy as _dg
from .derivative import cluster_matrices, dlams_dp
from .polygon import gauge_vector
from .optimize import OptimizeResult

__all__ = ['local_model', 'trust_region_step', 'maximize_ratio', 'model_gain']


def _active_cluster(lams, k, eps_abs):
    """Indices `[lo, hi)` of eigenvalues within `eps_abs` (absolute) of `lam_k`.

    An EPS-ACTIVE set rather than an exact-degeneracy test. The gap closes continuously as the
    optimizer approaches, so a rule that only fires at an exact tie would switch models one step
    before the end and never get the benefit; and one that fires too eagerly spends solves on
    cluster matrices it does not need. See the module docstring for how `eps_abs` is chosen.
    """
    lams = np.asarray(lams, dtype=float)
    lo = hi = k
    while lo > 0 and (lams[k] - lams[lo - 1]) <= eps_abs:
        lo -= 1
    while hi + 1 < len(lams) and (lams[hi + 1] - lams[k]) <= eps_abs:
        hi += 1
    return lo, hi + 1


def _cluster_scale(Ms):
    """`||(||M^(i)||_2)_i||` -- how far a unit step can move the cluster members apart."""
    return float(np.linalg.norm([np.linalg.norm(M, 2) for M in Ms]))


def local_model(family, p, solver, lams, k=1, eps_rel=1e-3, eps_abs=None, align_rtol=1e-9):
    """The first-order model of `lam_k/lam_1` at `p`.

    Returns a dict with `mode` ('smooth' or 'cluster'), the pieces the trust-region step needs,
    and the cluster range so a caller can see the model switch happening. `eps_abs` overrides
    `eps_rel*|lam_k|` when given, which is how `maximize_ratio` ties the active set to the trust
    radius.
    """
    lams = np.asarray(lams, dtype=float)
    lam_1, lam_k = float(lams[0]), float(lams[k])
    if eps_abs is None:
        eps_abs = float(eps_rel)*abs(lam_k)
    lo, hi = _active_cluster(lams, k, float(eps_abs))
    m, j = hi - lo, k - lo
    out = dict(cluster=(lo, hi), m=m, j=j, lam_1=lam_1, lam_k=lam_k, ratio=lam_k/lam_1,
               eps_abs=float(eps_abs))

    if m == 1:
        J = dlams_dp(family, p, solver, [lam_1, lam_k]) if k != 0 else \
            dlams_dp(family, p, solver, [lam_1])
        g1, gk = J[0], J[-1]
        out.update(mode='smooth', g1=g1, gk=gk,
                   grad=(gk*lam_1 - lam_k*g1)/lam_1**2,
                   scale=float(np.linalg.norm(gk)))
        return out

    g1 = dlams_dp(family, p, solver, [lam_1])[0]
    split = float(lams[hi - 1] - lams[lo])
    aligned = split > align_rtol*abs(lam_k)
    # Aligned basis when the split is resolvable, so that `Lam` and `Ms` live in the SAME basis;
    # the cluster basis with `Lam = 0` when it is not, where any orthonormal basis serves.
    Ms = cluster_matrices(family, p, solver, float(np.mean(lams[lo:hi])), m,
                          lams=(lams[lo:hi] if aligned else None))
    Lam = np.diag(lams[lo:hi] - lam_k) if aligned else np.zeros((m, m))
    out.update(mode='cluster', g1=g1, Ms=Ms, Lam=Lam, split=split, aligned=aligned,
               scale=_cluster_scale(Ms))
    return out


def model_gain(model, d):
    """The EXACT local model `dF(d)`, whatever the position `j` is.

    Every branch of `trust_region_step` reports its gain through this rather than through its own
    surrogate's optimal value. The two agree when the surrogate is exact (j = 0 and j = m-1), and
    when it is not -- the interior case -- this is the quantity the trust-region ratio has to be
    measured against, or `rho` compares a real improvement with a gain no direction delivers.
    """
    d = np.asarray(d, dtype=float)
    if model['mode'] == 'smooth':
        return float(model['grad'] @ d)
    S = model['Lam'] + np.einsum('i,ijk->jk', d, model['Ms'])
    sig = float(np.linalg.eigvalsh(S)[model['j']])
    lam_1, lam_k = model['lam_1'], model['lam_k']
    return float((sig*lam_1 - lam_k*(model['g1'] @ d))/lam_1**2)


def _step_smooth(model, delta):
    g = model['grad']
    n = float(np.linalg.norm(g))
    if n == 0:
        return np.zeros_like(g)
    return delta*g/n


def _step_concave_2x2(model, delta):
    """`lam_min` of a 2x2 block, as LINEAR MINUS A NORM rather than as an LMI.

    Writing `alpha = A.d + l0`, `gamma = C.d + l1`, `beta = B.d`,

        lam_min = (alpha+gamma)/2 - ||( (alpha-gamma)/2, beta )||

    so the subproblem is `max_{||d||<=Delta} [ a.d + c0 - lam_1*||L d + b|| ] / lam_1^2` -- a
    smooth concave program in `d` alone, with no auxiliary `t` and no determinant constraint.

    THAT REFORMULATION IS NOT COSMETIC. The previous version introduced `t` and imposed
    `tr >= 0`, `det >= 0`; at the natural start `z = 0` the determinant constraint is
    `l0*l1 = 0` whenever the cluster member is exactly at `lam_k` -- which is every iterate,
    since `j = 0` means `l0 = 0`. The constraint gradient degenerates there and SLSQP stalls at
    the start point. Measured against a 400-start brute-force maximization of the exact model:
    the LMI form returned 0.00727 where the true optimum was 0.01324, **45 per cent short**, at
    `Delta = 0.05` -- and it got worse as the radius shrank, which is exactly the regime a
    converging run spends its time in. This form agrees with brute force to 1e-13 relative, and
    where it differs it is the LARGER of the two.
    """
    Ms, g1, Lam = model['Ms'], model['g1'], model['Lam']
    lam_1, lam_k = model['lam_1'], model['lam_k']
    A, B, C = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    l0, l1 = float(Lam[0, 0]), float(Lam[1, 1])

    a = ((A + C)/2)*lam_1 - lam_k*g1
    c0 = (l0 + l1)/2*lam_1
    L = np.vstack([(A - C)/2, B])
    b = np.array([(l0 - l1)/2, 0.0])

    def neg(d):
        return -((a @ d + c0) - lam_1*np.linalg.norm(L @ d + b))/lam_1**2

    def neg_grad(d):
        w = L @ d + b
        n = float(np.linalg.norm(w))
        gw = (L.T @ w/n) if n > 0 else np.zeros_like(d)
        return -(a - lam_1*gw)/lam_1**2

    cons = [dict(type='ineq', fun=lambda d: delta**2 - d @ d, jac=lambda d: -2*d)]
    na = float(np.linalg.norm(a))
    starts = [np.zeros(len(g1))] + ([delta*a/na] if na > 0 else [])
    best, best_d = np.inf, np.zeros(len(g1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d0 in starts:
            res = minimize(neg, d0, jac=neg_grad, method='SLSQP', constraints=cons,
                           options=dict(maxiter=300, ftol=1e-16))
            d = np.asarray(res.x, dtype=float)
            nd = float(np.linalg.norm(d))
            if nd > delta:
                d = d*(delta/nd)
            if neg(d) < best:
                best, best_d = neg(d), d
    return best_d


def _step_concave_sdp(model, delta):
    """`lam_min` of an m x m block, m >= 3: the SDP itself, through `cvxpy`.

    Imported here rather than at module scope so that the m = 2 path -- which `D_N` symmetry
    makes the common one -- never needs the dependency installed.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:                                    # pragma: no cover
        raise NotImplementedError(
            f"a cluster of size {model['m']} needs the semidefinite program in the module "
            f"docstring, and cvxpy is not installed (`uv add cvxpy`). Only m = 2 reduces to "
            f"scalar trace/determinant constraints.") from exc

    Ms, g1, Lam = model['Ms'], model['g1'], model['Lam']
    lam_1, lam_k, m, n_p = model['lam_1'], model['lam_k'], model['m'], len(model['g1'])
    d = cp.Variable(n_p)
    t = cp.Variable()
    S = Lam + sum(d[i]*Ms[i] for i in range(n_p))
    prob = cp.Problem(cp.Maximize((t*lam_1 - lam_k*(g1 @ d))/lam_1**2),
                      [S - t*np.eye(m) >> 0, cp.norm(d) <= delta])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prob.solve()
    if d.value is None:
        return np.zeros(n_p)
    return np.asarray(d.value, dtype=float)


def _step_convex(model, delta, rng=None):
    """`lam_max` of an m x m block, by the sphere reduction in the module docstring.

    `G(u) = [u^T Lam u * lam_1 + Delta*||w(u)||]/lam_1^2` with `w(u) = m(u) lam_1 - lam_k g1`,
    maximized over `||u|| = 1`; the step is then `d = Delta w/||w||`. Exact, and the search is
    over `S^{m-1}` rather than over `R^{n_p}` -- one angle when m = 2.
    """
    Ms, g1, Lam = model['Ms'], model['g1'], model['Lam']
    lam_1, lam_k, m = model['lam_1'], model['lam_k'], model['m']

    def w_of(u):
        return np.einsum('ijk,j,k->i', Ms, u, u)*lam_1 - lam_k*g1

    def G(u):
        return (float(u @ Lam @ u)*lam_1 + delta*float(np.linalg.norm(w_of(u))))/lam_1**2

    if m == 2:
        # G(theta) has period pi (u and -u give the same u^T . u), so scan a half turn.
        thetas = np.linspace(0.0, np.pi, 181, endpoint=False)
        vals = [G(np.array([np.cos(t), np.sin(t)])) for t in thetas]
        i = int(np.argmax(vals))
        lo, hi = thetas[i] - np.pi/180, thetas[i] + np.pi/180
        res = minimize_scalar(lambda t: -G(np.array([np.cos(t), np.sin(t)])),
                              bounds=(lo, hi), method='bounded',
                              options=dict(xatol=1e-12))
        u = np.array([np.cos(res.x), np.sin(res.x)])
        if G(u) < vals[i]:
            u = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        starts = [np.eye(m)[i] for i in range(m)]
        starts += list(np.linalg.eigh(Lam)[1].T)
        starts += list(rng.normal(size=(10*m, m)))
        best, u = -np.inf, np.eye(m)[0]
        for s in starts:
            s = np.asarray(s, dtype=float)
            nrm = np.linalg.norm(s)
            if nrm == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = minimize(lambda v: -G(v/np.linalg.norm(v)), s/nrm, method='Nelder-Mead',
                             options=dict(xatol=1e-10, fatol=1e-14, maxiter=400*m))
            v = r.x/np.linalg.norm(r.x)
            if G(v) > best:
                best, u = G(v), v

    w = w_of(u)
    nw = float(np.linalg.norm(w))
    if nw == 0:
        return np.zeros(len(g1))
    return delta*w/nw


def _step_interior(model, delta, rng=None, n_rand=8):
    """`sigma_j` for 0 < j < m-1: neither convex nor concave, so maximize the EXACT model.

    There is no reduction here -- an interior ordered eigenvalue of an affine matrix function is
    neither, and the subproblem is genuinely nonconvex. But it is also genuinely CHEAP: one
    evaluation of `model_gain` is an `m x m` `eigvalsh` with `m` at most a handful, against ~300
    ms for the eigensolve that produced `Ms`. So this maximizes the exact model directly, by
    SLSQP from the two extreme-branch steps plus `n_rand` random directions on the trust sphere.

    The first version of this bracketed `sigma_j` between `sigma_0` and `sigma_{m-1}` and took
    the better of the two steps. Measured against a 200-start brute-force maximization, that was
    **10 to 43 per cent short** across m = 3, 4 -- a real loss, not a rounding one. Multi-start on
    the exact model agrees with the same reference to 1.3e-05 or better on every case tried
    (m = 3, 4, 5; j = 1, 2, 3; Delta = 0.01 and 0.3), and eight random starts is already enough:
    sixteen and twenty changed nothing.

    Still a heuristic, since a nonconvex maximization has no certificate. It is counted in
    `OptimizeResult.n_interior_steps` so a run says how much of itself rests on one.
    """
    n_p = len(model['g1'])
    rng = np.random.default_rng(0) if rng is None else rng
    seeds = [_step_convex(model, delta, rng=rng),
             _step_concave_2x2(model, delta) if model['m'] == 2
             else _step_concave_sdp(model, delta)]
    r = rng.normal(size=(n_rand, n_p))
    seeds += list(delta*r/np.linalg.norm(r, axis=1, keepdims=True))

    cons = [dict(type='ineq', fun=lambda d: delta**2 - d @ d, jac=lambda d: -2*d)]
    best, best_d = -np.inf, np.zeros(n_p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d0 in seeds:
            res = minimize(lambda d: -model_gain(model, d), d0, method='SLSQP',
                           constraints=cons, options=dict(maxiter=200, ftol=1e-14))
            d = np.asarray(res.x, dtype=float)
            nd = float(np.linalg.norm(d))
            if nd > delta:
                d = d*(delta/nd)
            g = model_gain(model, d)
            if g > best:
                best, best_d = g, d
    return best_d


def _active_rows(lin, delta):
    """Which linearized constraints a step of radius `delta` could actually reach.

    `c_i + A_i.d >= 0` can bind within the trust region only if `c_i < delta*||A_i||`. Most
    iterates are far from every constraint, and the exact branches above are exact only for the
    unconstrained ball -- so this is what decides whether to keep them or fall back.
    """
    c, A = lin
    if c.size == 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(c < delta*np.linalg.norm(A, axis=1) + 1e-15)


def _step_constrained(model, delta, lin, rng=None, n_rand=8):
    """Maximize the EXACT model over the trust region intersected with `c + A.d >= 0`.

    Used only when a constraint is within reach of the step, because none of the exact reductions
    survive the extra rows: the sphere reduction assumes the inner maximization runs over a plain
    ball, and the 2x2 concave form assumes the same. Rather than approximate each branch
    separately, this maximizes `model_gain` itself -- an `m x m` eigvalsh, free next to the
    eigensolve -- from the unconstrained branch's step plus random feasible directions.
    """
    c, A = lin
    n_p = len(model['g1']) if model['mode'] == 'cluster' else len(model['grad'])
    rng = np.random.default_rng(0) if rng is None else rng

    cons = [dict(type='ineq', fun=lambda d: delta**2 - d @ d, jac=lambda d: -2*d),
            dict(type='ineq', fun=lambda d: c + A @ d, jac=lambda d: A)]
    r = rng.normal(size=(n_rand, n_p))
    seeds = [np.zeros(n_p)] + list(0.5*delta*r/np.linalg.norm(r, axis=1, keepdims=True))

    best, best_d = -np.inf, np.zeros(n_p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d0 in seeds:
            res = minimize(lambda d: -model_gain(model, d), d0, method='SLSQP',
                           constraints=cons, options=dict(maxiter=200, ftol=1e-14))
            d = np.asarray(res.x, dtype=float)
            nd = float(np.linalg.norm(d))
            if nd > delta:
                d = d*(delta/nd)
            if np.min(c + A @ d) < -1e-9*max(1.0, float(np.abs(c).max())):
                continue                       # SLSQP returned something infeasible; discard
            g = model_gain(model, d)
            if g > best:
                best, best_d = g, d
    return best_d


def trust_region_step(model, delta, rng=None, lin=None):
    """`(d, gain)` maximizing the local model over `||d|| <= delta`, and over `c + A.d >= 0`.

    Dispatches on the position `j` of `lam_k` within its cluster: linear when the cluster is a
    singleton, concave when `lam_k` is its lowest member, convex when it is the highest, and a
    direct multi-start on the exact model when it is neither. The returned gain is always
    `model_gain(model, d)`, the exact model value at the step actually taken.

    `lin = (c, A)` linearizes a `ConstraintSet` at the current iterate. When no constraint is
    within reach of a step of this radius the exact branches run unchanged, which is the usual
    case; when one is, `_step_constrained` takes over, because none of the reductions survive the
    extra rows.
    """
    if lin is not None and len(_active_rows(lin, delta)):
        d = _step_constrained(model, delta, lin, rng=rng)
        gain = model_gain(model, d)
        return (d, float(gain)) if np.isfinite(gain) and gain > 0 else (np.zeros_like(d), 0.0)

    if model['mode'] == 'smooth':
        d = _step_smooth(model, delta)
        return d, max(0.0, model_gain(model, d))

    m, j = model['m'], model['j']
    if j == 0:
        d = _step_concave_2x2(model, delta) if m == 2 else _step_concave_sdp(model, delta)
    elif j == m - 1:
        d = _step_convex(model, delta, rng=rng)
    else:
        d = _step_interior(model, delta, rng=rng)

    d = np.asarray(d, dtype=float)
    nd = float(np.linalg.norm(d))
    if not np.all(np.isfinite(d)):
        return np.zeros(len(model['g1'])), 0.0
    if nd > delta:                       # SLSQP satisfies the trust region to its own tolerance
        d = d*(delta/nd)
    gain = model_gain(model, d)
    if not np.isfinite(gain) or gain < 0:
        return np.zeros(len(model['g1'])), 0.0
    return d, float(gain)


def _degeneracy_move(evaluator, p, F, k, gain_tol=0.0):
    """Try crossing to the other representation of a nearly-degenerate shape.

    TWO MOVES, ONE MECHANISM. Both cross a discontinuity in the parametrization that no local
    step can cross, and both are proposed, evaluated and accepted only on a measured improvement.

    FLATTEN sets a nearly-flat vertex exactly flat. `F` has a concave KINK there -- one-sided
    derivatives +3.414 and -1.234 on the two sides of the sqrt(8/3) rectangle's flat vertex, with
    `F(0) = 35/11` strictly above both neighbours -- so a first-order model cannot land on it and
    a trust region grinds its radius to nothing approaching it. See `degeneracy.flatten_candidates`.

    COLLAPSE removes a short edge. A short edge and a flat vertex describe nearly the same domain but
    sit in completely different regions of `p`-space, and no local step connects them. So an
    optimizer that wants an edge gone presses against the `MinEdgeLength` floor and stops, with
    the domain it wanted freely available elsewhere in the family. Measured on `max lam_3/lam_1`
    at N=5, from two starts: both pinned the fifth edge at the floor to a slack of 1e-16 and
    stalled at 2.973 and 3.024, where the quadrilateral answer is 35/11 = 3.182.

    THE MOVE IS A TRIAL STEP, NOT A REWRITE. Deleting an edge of length `eps` moves `lam` by
    `O(eps)`, so this is not domain-preserving and cannot be made so at any useful threshold.
    That is fine: it is proposed, evaluated, and accepted only if the objective actually improves
    -- the same accept/reject test every other step faces. Both endpoints of the short edge are
    tried, since which to keep is not obvious in advance.

    Returns `(p, F, lams, solver)` for the better accepted candidate, or `None`.
    """
    fam = evaluator.family
    try:
        v = fam.vertices(p)
        # FLATTEN first: it is the move that lands ON the kink, and it is exact in the sense that
        # matters -- the flat configuration is a genuine strict local maximum of F, not an
        # approximation to one. COLLAPSE is the cruder fallback for a shape pinned at the edge
        # floor, where there is nothing to flatten.
        cands = _dg.flatten_candidates(v, perim=fam.perim)
        cands += _dg.collapse_candidates(v, perim=fam.perim)
    except (ValueError, TypeError):
        return None
    if not cands:
        return None

    best, plan_at = None, evaluator._plan_at
    for v in cands:
        try:
            q = gauge_vector(v, fam.perim, anchor=0)
        except (ValueError, TypeError):
            continue
        if not fam.is_valid(q):
            continue
        try:
            # REPLAN, not merely reseed. The candidate has one corner FEWER than the shape the
            # plan was frozen on -- a flat vertex is not a corner -- and `basis_plan.realize`
            # raises IndexError on the mismatch. A discrete jump is the one place re-planning is
            # legitimate, since the objective moves by O(eps) across it anyway.
            evaluator.replan(q)
            lams_q, solver_q = evaluator.eigenvalues(q)
        except Exception as exc:
            # NOT silent. A bare `except: continue` here hid exactly the IndexError above through
            # three full runs, each reporting `n_degeneracy_moves=0` as though the move had been
            # offered and declined on merit.
            warnings.warn(f'degeneracy move: candidate rejected by {type(exc).__name__}: {exc}')
            continue
        F_q = float(lams_q[k]/lams_q[0])
        if F_q > F + gain_tol and (best is None or F_q > best[1]):
            best = (q, F_q, lams_q, solver_q)

    # `_plan` currently holds whichever candidate was tried LAST, which need not be the winner.
    # Re-freeze it lazily at the shape actually being returned to, or restore the run's own.
    evaluator.replan(plan_at if best is None else best[0])
    return best


def maximize_ratio(evaluator, p0, k=1, delta0=0.05, delta_max=0.5, eps_rel='auto',
                   max_iter=200, gain_tol=1e-7, eta=0.1, callback=None, rng=None,
                   verify_final=True, constraints=None, degeneracy_moves=True):
    """Trust-region ascent on `lam_k/lam_1`, valid through a degeneracy.

    `evaluator` must be a `SpectrumEvaluator` with `degenerate_tol=0` -- the guard that normally
    refuses a merged spectrum has to be off, because handling it is the point here.

    `eps_rel='auto'` sizes the eps-active cluster from the trust radius, which is the principled
    choice (module docstring); a float pins it to `eps_rel*|lam_k|` as it used to be.

    `constraints` is a `douse.constraints.ConstraintSet`. It is linearized into the trust-region
    subproblem, so the step is steered along the feasible boundary rather than merely rejected at
    it, and trial points are checked before an eigensolve is spent on them. A
    `MinEdgeLength` here is a gauge fixing rather than a restriction -- see that module.

    `degeneracy_moves` lets a run cross from a short edge to a flat vertex when it is pinned
    against the edge floor, which no local step can do -- see `_degeneracy_move`. It costs two
    eigensolves, is tried only when the floor is binding AND the ordinary step was rejected, and
    is accepted only on a measured improvement.

    Stops when the model's best first-order gain per unit radius falls below `gain_tol`, which is
    the first-order optimality certificate, not a step-size heuristic.

    `verify_final` re-solves the last iterate from cold before returning, and IT IS NOT OPTIONAL
    IN SPIRIT. These runs converge ONTO a degeneracy, which is the one place tracking is least
    reliable: two seeds land on one merged value, `_verify_multiplicity` sees a legitimate-looking
    double and passes it, and the reported ratio is off by more than the run's own claimed
    accuracy. Measured on the N=4 run that converges to the square: the tracked endpoint reported
    2.500000005354, which is 5.4e-09 ABOVE 5/2 -- impossible if the square is the maximizer, and
    the kind of number that ends up in a table. A cold solve at the very same shape gives
    2.499999994645, 5.4e-09 below, and the shape is a square to 2.1e-06 in the sides. The
    discretization was never the problem; the continuation was. One extra solve per RUN is
    nothing, and `lams_tracked`/`fun_tracked` keep the pre-verification values so the size of the
    discrepancy stays visible.
    """
    if evaluator.degenerate_tol > 0:
        raise ValueError('maximize_ratio handles degeneracies itself; construct the evaluator '
                         'with degenerate_tol=0 so it does not refuse them first')
    fam = evaluator.family
    p = np.array(p0, dtype=float)
    delta, message, success = float(delta0), f'reached max_iter={max_iter}', False
    n_cluster_steps, n_interior_steps, n_infeasible, it = 0, 0, 0, 0
    n_degeneracy_moves = 0
    auto = isinstance(eps_rel, str)
    if auto and eps_rel != 'auto':
        raise ValueError(f"eps_rel must be a float or 'auto', got {eps_rel!r}")
    scale = None                      # bootstrapped from the first model

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
            eps_abs = None if (not auto or scale is None) else delta*scale
            model = local_model(fam, p, solver, lams, k=k,
                                eps_rel=(1e-3 if auto else eps_rel), eps_abs=eps_abs)
            scale = model['scale']
            if model['mode'] == 'cluster':
                n_cluster_steps += 1
                if 0 < model['j'] < model['m'] - 1:
                    n_interior_steps += 1

            lin = None if constraints is None else constraints(p)
            d, gain = trust_region_step(model, delta, rng=rng, lin=lin)
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
            rho = -np.inf
            # A step can fail two ways and BOTH have to reach the degeneracy move below. An
            # earlier version `continue`d on an infeasible trial, so the move was only ever
            # offered after a feasible-but-unproductive step -- and against an active edge floor
            # the linearized step is usually infeasible instead (measured: 57 of 83 iterations),
            # so the move never fired once in either N=5 run.
            if not fam.is_valid(q) or (constraints is not None
                                       and not constraints.is_feasible(q)):
                delta *= 0.5
                n_infeasible += 1
            else:
                lams_q, solver_q = evaluator.eigenvalues(q)
                F_q = float(lams_q[k]/lams_q[0])
                rho = (F_q - F)/gain if gain > 0 else -np.inf
                if rho > eta:
                    p, F, lams, solver = q, F_q, lams_q, solver_q
                    if rho > 0.75:
                        delta = min(2*delta, delta_max)
                else:
                    delta *= 0.5
                    # The tracker's continuation now points at the rejected trial; reseed it from
                    # the accepted iterate so the next evaluation continues from where we are.
                    evaluator._last = (p.copy(), np.asarray(lams).copy())

            # Pinned against the edge floor with the step rejected: the shape wants to degenerate
            # and cannot get there by moving. Offer it the other representation.
            # Trigger on EITHER a nearly-flat vertex (there is a kink to land on) or a binding
            # edge floor (there is a representation to cross to). Requiring a binding constraint
            # for both, as an earlier version did, meant the flatten move could never fire: a
            # shape approaching a kink is nowhere near the edge floor.
            near_kink = False
            if degeneracy_moves and rho <= eta and delta < 0.05*delta0:
                try:
                    near_kink = bool(len(_dg.report(fam.vertices(p)).near_flat_idx))
                except (ValueError, TypeError):
                    near_kink = False
            pinned = (constraints is not None and constraints.worst(p)[1] < 1e-9)
            if (rho <= eta and degeneracy_moves and delta < 0.05*delta0
                    and (near_kink or pinned)):
                moved = _degeneracy_move(evaluator, p, F, k)
                if moved is not None:
                    p, F, lams, solver = moved
                    delta = float(delta0)
                    n_degeneracy_moves += 1
                else:
                    degeneracy_moves = False       # it was offered and refused; do not re-pay
                    evaluator._last = (p.copy(), np.asarray(lams).copy())
            if callback is not None:
                callback(p, F, model, delta, rho)
            if delta < 1e-14:
                message = f'trust radius collapsed to {delta:.3g}'
                break

    lams_tracked, F_tracked = np.asarray(lams).copy(), F
    if verify_final:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evaluator.reset_continuation()
            lams, _ = evaluator.eigenvalues(p)
        F = float(lams[k]/lams[0])

    gap = (float(np.min(np.diff(np.sort(lams))/np.sort(lams)[:-1]))
           if len(lams) > 1 else np.inf)
    return OptimizeResult(x=p, fun=F, nit=it, lams=lams, gap=gap, delta=delta,
                          n_cluster_steps=n_cluster_steps, n_interior_steps=n_interior_steps,
                          n_infeasible=n_infeasible,
                          n_degeneracy_moves=n_degeneracy_moves,
                          # THE BASIS HEALTH GOES IN THE RESULT, always. A run that reports a
                          # ratio without saying how well its basis was resolving is exactly what
                          # produced 3.0717 for a problem whose answer is above 3.18: the frozen
                          # plan had drifted to sigma = 1.6e-04 and nothing said so. These two
                          # numbers make that visible on the first run instead of the fourth.
                          worst_sigma=float(getattr(evaluator, 'worst_sigma', np.nan)),
                          n_replans=int(getattr(evaluator, 'n_replans', 0)),
                          lams_tracked=lams_tracked, fun_tracked=F_tracked,
                          verified=bool(verify_final), success=success, message=message)
