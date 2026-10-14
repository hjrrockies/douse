"""`max lam_3/lam_1` over N-gons: the convex branch, and what it finds.

    uv run python -m benchmarks.ratio_k3 [--N 4 5] [--max-iter 250] [--no-constraints]

WHY k=3 IS THE INTERESTING ONE. At a regular N-gon `lam_2` is the LOWEST member of its double and
`lam_3` is the HIGHEST, so `d(lam_3)` is `lam_max` of an affine matrix function -- convex, and the
old `cluster_opt` refused it outright on the grounds that maximizing a convex function has no
unique local-model optimum. Exchanging the two maxima removes the obstacle exactly (see
`cluster_opt._step_convex`), and the first thing the branch says is that the conjectured shape is
wrong: at the exact square the model gain for `lam_2/lam_1` is 0, as it must be, but for
`lam_3/lam_1` it is 4.4e-02.

N=4: THE ANSWER IS 35/11, EXACTLY. Both starts converge to the RECTANGLE with `L/H = sqrt(8/3)`,
all angles 90 degrees, reached to 1.2e-09 and 1.6e-09. There the (1,2) and (3,1) modes cross --
`1/L^2 + 4/H^2 = 9/L^2 + 1/H^2`, i.e. `L^2/H^2 = 8/3` -- and with the perimeter fixed at 4,

    lam_2/lam_1 = 20/11 = 1.818181...        lam_3/lam_1 = 35/11 = 3.181818...

against 5/2 at the square. The trajectory crosses branches: `lam_3` starts as the highest of
`{lam_2, lam_3}` and ends as the LOWEST of a new cluster `{lam_3, lam_4}`, so the run finishes in
the concave solver -- but only the convex branch could say the square was worth leaving.

N=5: A GENUINE PENTAGON BEATS IT, and the history here is a warning. Runs with the basis plan
frozen at the starting shape stalled at 3.072 and 3.078 carrying a vertex 1.4 degrees from flat,
which read convincingly as degeneration toward the quadrilateral. It was not. The frozen plan had
drifted to `sigma = 1.6e-04` where a plan built at that shape gives `5.4e-11` -- a factor of 2.9
million -- putting 1.6e-06 of error into `F`, which is exactly the scale at which the trust
region's `rho` goes negative and the radius collapses. With `SpectrumEvaluator(replan_tol=...)`
watching the tension, both starts instead walk AWAY from the near-flat vertex to non-degenerate
pentagons:

    from a near-regular start   F = 3.191229   (+9.4e-03 over 35/11)
    from a generic start        F = 3.187127   (+5.3e-03)

Both `reached max_iter=250`, so **both are lower bounds**, and their shapes differ visibly
(angles 115/125/91/92/117 against 94/101/92/90/163) -- one optimum or two is open.

READ `worst_sigma` IN THE OUTPUT. It is printed for every run precisely because ignoring it is
what produced the retracted N=5 result above.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from douse import (ClosingBudget, ConstraintSet, MinEdgeLength, SpectrumEvaluator,
                   VertexPolygon, param_vector)
from douse import degeneracy as dg
from douse.cluster_opt import maximize_ratio

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_k3')
EXACT_QUAD = 35/11


def regular(N, perim=PERIM):
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


def start_shape(N, jitter, seed):
    rng = np.random.default_rng(seed)
    v = regular(N)
    return v + jitter*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()


STARTS = (('from_regular', 0.02, 3), ('from_generic', 0.10, 11))


def run_one(N, label, jitter, seed, max_iter=250, use_constraints=True, replan_tol=1e-9):
    fam = VertexPolygon(N, perim=PERIM)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam)) if use_constraints else None
    ev = SpectrumEvaluator(fam, n_eigs=4, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=8, replan_tol=replan_tol)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(start_shape(N, jitter, seed), PERIM), k=2,
                             max_iter=max_iter, constraints=cons)
    v = fam.vertices(res.x)
    rep = dg.report(v)
    return res, rep, v, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='*', default=[4, 5])
    ap.add_argument('--max-iter', type=int, default=250)
    ap.add_argument('--no-constraints', action='store_true')
    ap.add_argument('--replan-tol', type=float, default=1e-9)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f'quadrilateral answer: 35/11 = {EXACT_QUAD:.12f}\n')
    for N in args.N:
        for label, jitter, seed in STARTS:
            res, rep, v, secs = run_one(N, label, jitter, seed, args.max_iter,
                                        not args.no_constraints, args.replan_tol)
            print(f'N={N} k=3 {label}: F = {res.fun:.12f}   (35/11 {res.fun-EXACT_QUAD:+.3e})')
            print(f'    nit={res.nit}/{args.max_iter} {secs:.0f}s  moves={res.n_degeneracy_moves} '
                  f'replans={res.n_replans}  worst_sigma={res.worst_sigma:.1e}')
            print(f'    lams   = {np.array2string(np.asarray(res.lams), precision=8)}')
            print(f'    edges  = {np.array2string(rep.edge_lengths, precision=6)}')
            print(f'    angles = {np.array2string(np.degrees(dg.interior_angles(v)), precision=3)}')
            print(f'    {rep}')
            print(f'    {res.message}\n', flush=True)
            json.dump(dict(N=N, label=label, fun=float(res.fun), nit=int(res.nit),
                           x=[float(t) for t in res.x],
                           lams=[float(t) for t in np.asarray(res.lams)],
                           worst_sigma=float(res.worst_sigma), n_replans=int(res.n_replans),
                           n_degeneracy_moves=int(res.n_degeneracy_moves),
                           message=str(res.message), seconds=round(secs, 1)),
                      open(os.path.join(OUT, f'N{N}_{label}.json'), 'w'))


if __name__ == '__main__':
    main()
