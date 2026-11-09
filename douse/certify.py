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

from .derivative import (cluster_matrices, detect_cluster, dlams_dp, directional_derivatives,
                         whole_clusters)

__all__ = ['det_form', 'first_order_report', 'second_order_probe', 'antisymmetric_hessian']


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

        cluster        (lo, hi) indices of the cluster containing lam_k: the values within
                       `cluster_rtol` of each other, widened to whole chladni clusters
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
    # chladni's clusters are a floor under the gap test, as for the optimizer's eps-active set:
    # a block that cuts one is refused by the backend (`UnresolvedSpectrum`)
    lo, hi = whole_clusters(getattr(solver, 'spec', None), lo, hi)
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


def antisymmetric_hessian(F, vertices, basis, ts=(0.01, 0.03)):
    """Is a mirror-symmetric optimum a maximum among ALL polygons, not only symmetric ones?

    `F(vertices) -> float` is the objective on vertex arrays (complex); `basis` is
    `MirrorPolygon.antisymmetric_basis`, `(2N, r)` with rows `[x.., y..]`. Along a direction `d`
    in its span the reflected polygon of `v + t d` is congruent to `v - t d`, so `F(v + t d) =
    F(v - t d)`: the first derivative vanishes and ONE evaluation per direction gives the
    curvature `D(d) = 2 (F(v + t d) - F(v)) / t^2`. The quadratic form on the span is read off
    `D(e_i)` and `D(e_i + e_j)`: `r (r + 1)/2` evaluations per step size.

    TWO STEP SIZES, BECAUSE THE OPTIMUM MAY BE A KINK. When `lam_k` there is a double eigenvalue
    with one even and one odd eigenfunction, an antisymmetric perturbation splits it at FIRST
    order, and `F(v + t d) - F(v)` is `-c|t|`, not `-c t^2`: `D` then scales like `1/t`. The
    ratio `D(t_0)/D(t_1)` per direction tells the two apart (1 for smooth, `t_1/t_0` for a
    kink). Either way a maximum needs every `D <= 0`; one positive value refutes it. Like
    `second_order_probe`, this is evidence from finitely many steps, not a certificate.

    Returns `dict(F0, ts, D (len(ts), r, r) of the sampled curvatures [D(e_i) on the diagonal,
    D(e_i + e_j)/2 off it], hess_eigs (len(ts), r) of the implied quadratic form,
    worst (max D), scaling (D(t_0)/D(t_1) per basis direction), verdict)`.
    """
    v = np.asarray(vertices, dtype=complex)
    n = len(v)
    B = np.asarray(basis, dtype=float)
    r = B.shape[1]
    F0 = float(F(v))
    Ds, eigs, worst = [], [], -np.inf
    n_done, n_probes = 0, 0
    for t in ts:
        D = np.full((r, r), np.nan)
        for i in range(r):
            for j in range(i, r):
                d = B[:, i] + (B[:, j] if j != i else 0.0)
                dv = d[:n] + 1j*d[n:]
                try:
                    D[i, j] = D[j, i] = 2*(float(F(v + t*dv)) - F0)/t**2/(1 if i == j else 2)
                except Exception:                                    # noqa: BLE001
                    pass
        worst = max(worst, np.nanmax(D)) if np.isfinite(D).any() else worst
        n_done += int(np.isfinite(D[np.triu_indices(r)]).sum())
        n_probes += len(D[np.triu_indices(r)])
        diag = np.diag(D).copy()
        H = D - (diag[:, None] + diag[None, :])/2                   # (D(ei+ej) - D(ei) - D(ej))/2
        H[np.diag_indices(r)] = diag
        eigs.append(np.linalg.eigvalsh(H) if np.isfinite(H).all() else np.full(r, np.nan))
        Ds.append(D)
    Ds, eigs = np.array(Ds), np.array(eigs)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.diagonal(Ds[0])/np.diagonal(Ds[-1])
    ok = np.isfinite(worst)
    # A probe that refused says nothing, so a "local maximum" needs most of them to have answered.
    enough = n_done >= 0.75*n_probes
    return dict(F0=F0, ts=list(ts), D=Ds, hess_eigs=eigs, worst=float(worst) if ok else None,
                scaling=scaling, n_done=n_done, n_probes=n_probes,
                verdict=('no evaluations' if not ok else
                         'refuted: F increases along a symmetry-breaking direction' if worst > 0 else
                         f'inconclusive: only {n_done} of {n_probes} probes were served' if not enough else
                         'consistent with a local maximum among all polygons' if worst <= 0 else
                         'refuted: F increases along a symmetry-breaking direction'))
