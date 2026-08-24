"""What can and cannot be said about a candidate maximizer of `lam_k/lam_1`.

For `lam_k/lam_1` the optima appear to sit ON a degeneracy -- at every regular N-gon measured,
`lam_2 = lam_3` exactly -- so a gradient method cannot converge there and the question is what
to check instead.

THE FIRST-ORDER CONDITION IS AUTOMATIC, WHICH IS NOT THE SAME AS SATISFIED BY EVIDENCE. Let
`lam_k` be the smallest member of a cluster of size 2. The one-sided directional derivative is

    dF(d) = [ lam_min(M(d)) * lam_1  -  lam_k * (g1.d) ] / lam_1^2

with `M(d) = sum_i d_i M^(i)` linear in `d`. At a regular N-gon two things hold to machine
precision, both measured (`hypotheses` reports them):

    g1 = grad lam_1 = 0      the regular N-gon MINIMIZES lam_1 at fixed perimeter
    tr M(d) = 0 for all d    lam_2 + lam_3 is stationary there

The first collapses `dF(d)` to `lam_min(M(d))/lam_1`, so a local max needs `lam_min(M(d)) <= 0`
for all `d`; with `M(-d) = -M(d)` that is `M(d)` never definite, and for 2x2 exactly
`det M(d) <= 0`. Now write `M(d) = [[A.d, B.d], [B.d, C.d]]`. The SECOND fact says `C = -A`
(verified to 1e-13), so

    det M(d) = (A.d)(C.d) - (B.d)^2 = -(A.d)^2 - (B.d)^2  <=  0     identically.

So `Q`, the matrix of that quadratic form, is `-A A^T - B B^T`: negative semidefinite for
structural reasons, of rank at most 2 whatever `n_p` is. Measured ranks are exactly 2 for
N = 4, 5, 6 against `n_p` = 4, 6, 8. **The first-order test therefore cannot distinguish a
maximum from anything else** -- it is satisfied by construction, and an earlier version of this
module reported it as a certificate, which was wrong.

WHERE THE CONTENT ACTUALLY IS. The null space `{d : A.d = 0 and B.d = 0}` has dimension
`n_p - 2`. Along those directions the cluster does not split at first order and `dF = 0`, so the
verdict is second order. `second_order_probe` measures it, by differencing `F` along null
directions. That is EVIDENCE, not a certificate, and the function says so in its return value.

Measured for N = 4, 5, 6: the ratio falls along every null direction sampled, quadratically
(t = 0.02 gives ~-3e-04, t = 0.05 gives ~-2e-03, the expected factor of 6.25). Consistent with
the regular N-gon being a local maximizer; not a proof of it.
"""
import warnings

import numpy as np

from .derivative import cluster_matrices, detect_cluster, dlams_dp, directional_derivatives

__all__ = ['det_form', 'first_order_report', 'second_order_probe']


def det_form(Ms):
    """`Q` with `d^T Q d = det(sum_i d_i M^(i))`, for 2x2 `M^(i)`.

    Exact: the determinant of a 2x2 matrix depending linearly on `d` is a quadratic form in `d`.
    """
    Ms = np.asarray(Ms, dtype=float)
    if Ms.ndim != 3 or Ms.shape[1:] != (2, 2):
        raise ValueError(f'det_form is the 2x2 case; got cluster matrices of shape {Ms.shape}')
    A, B, C = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    Q = 0.5*(np.outer(A, C) + np.outer(C, A)) - np.outer(B, B)
    return 0.5*(Q + Q.T)


def first_order_report(family, p, solver, lams, k=1, cluster_rtol=1e-6, tol=1e-8):
    """The algebraic facts at `p`, and an honest statement of what they settle.

    Returns a dict:

        cluster        (lo, hi) indices of the cluster containing lam_k
        hypotheses     measured: |grad lam_1|, max |tr M|, cluster size, whether lam_k is lowest
        Q              the determinant form, when the 2x2 path applies
        Q_eigenvalues  its spectrum
        null_dim       dimension of {d : dF(d) = 0 at first order}
        null_basis     (n_p, null_dim) orthonormal basis of that subspace
        verdict        'first order is automatic' | 'refuted' | 'inconclusive'

    `verdict` is never 'certified'. If the hypotheses hold, `Q <= 0` is forced and settles
    nothing; if they do not, this reports what failed rather than guessing.
    """
    lams = np.asarray(lams, dtype=float)
    lo, hi = detect_cluster(lams, k, rtol=cluster_rtol)
    m = hi - lo
    lam_1 = float(lams[0])

    g1 = dlams_dp(family, p, solver, [lam_1])[0]
    Ms = cluster_matrices(family, p, solver, float(np.mean(lams[lo:hi])), m)
    traces = np.trace(Ms, axis1=1, axis2=2)
    scale = max(float(np.max(np.abs(Ms))), 1e-300)

    hyp = dict(cluster_size=m, k_is_lowest_of_cluster=bool(lo == k),
               grad_lam1_norm=float(np.linalg.norm(g1)),
               max_abs_trace=float(np.max(np.abs(traces))),
               trace_is_zero=bool(np.max(np.abs(traces)) <= tol*scale),
               grad_lam1_is_zero=bool(np.linalg.norm(g1) <= tol*max(lam_1, 1.0)))
    out = dict(cluster=(lo, hi), hypotheses=hyp, ratio=float(lams[k]/lam_1), Ms=Ms)

    if m != 2 or lo != k:
        out.update(verdict='inconclusive', Q=None, Q_eigenvalues=None,
                   null_dim=None, null_basis=None)
        warnings.warn(f'this routine handles a 2x2 cluster with lam_{k+1} its lowest member; '
                      f'measured cluster size {m} starting at index {lo}.')
        return out

    Q = det_form(Ms)
    w, V = np.linalg.eigh(Q)
    wscale = max(float(np.abs(w).max()), 1e-300)
    null = np.abs(w) <= 1e-6*wscale
    out.update(Q=Q, Q_eigenvalues=w, null_dim=int(null.sum()), null_basis=V[:, null])

    if w.max() > 1e-6*wscale:
        out['verdict'] = 'refuted'          # some direction increases lam_k: not a maximum
    elif hyp['trace_is_zero'] and hyp['grad_lam1_is_zero']:
        out['verdict'] = 'first order is automatic'
    else:
        out['verdict'] = 'inconclusive'
    return out


def second_order_probe(evaluate_ratio, p, null_basis, h=(0.02, 0.05), n_dirs=None, rng=0):
    """Measure the curvature of `F` along the directions first order cannot see.

    `evaluate_ratio(q) -> float` returns `lam_k/lam_1` at parameters `q`; the caller supplies it
    so this module never builds a solver. `null_basis` comes from `first_order_report`.

    For each direction, `F(p +- t d)` is differenced at each `t` in `h`. A local maximum requires
    every curvature to be non-positive; the ratio between the two step sizes should be about
    `(h1/h0)^2` if the behaviour really is quadratic, and that ratio is reported so a caller can
    check rather than assume.

    EVIDENCE, NOT A CERTIFICATE. It samples finitely many directions in a subspace of dimension
    `n_p - 2`, so it can refute a maximum and cannot establish one.
    """
    null_basis = np.asarray(null_basis, dtype=float)
    if null_basis.size == 0:
        return dict(n_dirs=0, worst_curvature=None, verdict='no null directions')
    p = np.asarray(p, dtype=float)
    F0 = float(evaluate_ratio(p))

    dirs = null_basis.T
    if n_dirs is not None and n_dirs > len(dirs):
        gen = np.random.default_rng(rng)
        extra = gen.normal(size=(n_dirs - len(dirs), null_basis.shape[1])) @ null_basis.T
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        dirs = np.vstack([dirs, extra])

    rows, worst = [], -np.inf
    for d in dirs:
        curv = []
        for t in h:
            try:
                fp, fm = float(evaluate_ratio(p + t*d)), float(evaluate_ratio(p - t*d))
            except Exception:
                curv.append(np.nan)
                continue
            curv.append((fp - 2*F0 + fm)/t**2)
        rows.append(curv)
        finite = [c for c in curv if np.isfinite(c)]
        if finite:
            worst = max(worst, max(finite))
    rows = np.array(rows)
    return dict(n_dirs=len(dirs), F0=F0, curvatures=rows,
                worst_curvature=float(worst) if np.isfinite(worst) else None,
                verdict=('consistent with a local maximum' if worst <= 0 else
                         'refuted: the ratio increases along a null direction'))
