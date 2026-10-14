"""How an N-gon degenerates to an M-gon: `lam_k/lam_1` has a KINK at a flat vertex.

    uv run python -m benchmarks.kink_sweep [--k 3] [--json out.json]

THE QUESTION. An N-gon loses a vertex two ways, reaching the SAME limiting domain: an edge
collapses, or a vertex FLATTENS onto the edge it sits on. `douse.constraints` argues that the
second is the numerically benign route and forbids only the first, which makes a minimum edge
length a gauge fixing rather than a restriction. That argument says nothing about what the
OBJECTIVE does as a vertex flattens, and an optimizer has to live with the answer.

THE SHAPE. The base is the `sqrt(8/3)` rectangle -- the exact maximizer of `lam_3/lam_1` over
quadrilaterals of fixed perimeter, where the (1,2) and (3,1) modes cross (`3/H^2 = 8/L^2`) and
`F = 35/11` exactly. It is written as a PENTAGON with a fifth vertex at the midpoint of a long
edge, displaced by `h` along the outward normal. `h = 0` is the quadrilateral; `h > 0` pushes the
vertex out (convex corner, `nu = pi/omega > 1`); `h < 0` pushes it in (reentrant, `nu < 1`, a
genuine corner singularity).

THE ANSWER, and it is the reason `degeneracy.flatten_candidates` exists:

    dF/dh -> +3.413956   as h -> 0 from the reentrant side
    dF/dh -> -1.234267   as h -> 0 from the convex side
    F(0)   = 3.181818181821   against 35/11 = 3.181818181818
    F(-1e-08) = 3.181818147679,  F(+1e-08) = 3.181818169475   -- BOTH BELOW F(0)

`F` is continuous, both one-sided derivatives are finite and nonzero, and they have opposite
signs: a concave KINK whose maximum sits exactly at the degeneracy. The corner exponent
`nu = pi/omega` crosses 1 there, so the two sides are genuinely different problems meeting at a
point.

WHAT THAT COSTS AN OPTIMIZER. No first-order model can land on a kink. A trust-region method
approaching one proposes a step of size `Delta`, overshoots, is rejected, halves, and repeats --
the radius collapses geometrically while the iterate creeps. This is the same pathology as the
eigenvalue degeneracy `douse.cluster_opt` exists for, except that this one is geometric rather
than spectral, and no cluster matrix models it. Hence a discrete `flatten` move: propose `h = 0`
exactly, evaluate, accept on a measured improvement.

THE NUMERICS ARE NOT THE STORY. Every sample here uses a FRESH basis plan, and the tension stays
at 1e-13 to 1e-12 across the whole sweep including `|h| = 1e-08` (1.7e-15 at `h = 0`). The kink
is a property of the problem, not of the discretization -- which matters, because a separate
finding in this work turned out to be entirely an artifact of a stale frozen plan.

A FROZEN PLAN CANNOT CROSS IT. `BasisPlan` carries one block per CORNER and a flat vertex is not
one, so a plan frozen at `h != 0` raises `IndexError` inside `basis_plan.realize` at `h = 0`
exactly -- while working fine at `|h| = 1e-08` either side. The last section of the sweep
demonstrates that; `douse.SpectrumEvaluator.replan` is the workaround.
"""
import argparse
import json
import warnings

import numpy as np

from lappy import Eigenproblem, Polygon, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

PERIM = 4.0
_T = np.sqrt(8/3)
H_RECT = 2/(1 + _T)
L_RECT = 2*_T/(1 + _T)

# |h| values sampled either side of zero, plus zero itself
HS = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-8])


def pentagon(h):
    """The `sqrt(8/3)` rectangle with a fifth vertex `h` off the midpoint of the bottom edge."""
    v = np.array([0, L_RECT/2 - 1j*h, L_RECT, L_RECT + 1j*H_RECT, 1j*H_RECT], dtype=complex)
    return v*PERIM/np.abs(v - np.roll(v, 1)).sum()


def solve_fresh(h, n_eigs, precision=1e-12):
    """`(lams, sigma, n_cols, n_corners)` with a plan built AT this shape."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dom = Polygon(pentagon(h), bc='dir')
        evp = Eigenproblem(dom, precision=precision)
        lams = np.asarray(evp.solve(n_eigs), dtype=float)
        sigma = float(np.atleast_1d(evp.eval_solver.sigma(float(lams[0])))[0])
        return lams, sigma, len(evp.eval_solver.basis), len(dom.corners)


def sweep(k=2, precision=1e-12):
    """`k` is 0-based, so k=2 is `lam_3/lam_1`. Returns a list of records, ascending in `h`."""
    hs = np.concatenate([-HS, [0.0], HS[::-1]])
    hs = np.sort(hs)
    out = []
    for h in hs:
        try:
            lams, sigma, ncol, ncorn = solve_fresh(float(h), k + 2, precision)
            out.append(dict(h=float(h), F=float(lams[k]/lams[0]), lam1=float(lams[0]),
                            sigma=sigma, n_cols=ncol, n_corners=ncorn,
                            lams=[float(x) for x in lams]))
        except Exception as exc:
            out.append(dict(h=float(h), error=f'{type(exc).__name__}: {exc}'))
    return out


def one_sided_slopes(rows):
    """`(left, right)` limits of dF/dh at h=0, from the two samples closest to it on each side."""
    ok = [r for r in rows if 'F' in r]
    zero = [r for r in ok if r['h'] == 0.0]
    if not zero:
        return None, None
    F0 = zero[0]['F']
    left = [r for r in ok if r['h'] < 0]
    right = [r for r in ok if r['h'] > 0]
    lo = max(left, key=lambda r: r['h']) if left else None
    hi = min(right, key=lambda r: r['h']) if right else None
    return ((F0 - lo['F'])/(0.0 - lo['h']) if lo else None,
            (hi['F'] - F0)/(hi['h'] - 0.0) if hi else None)


def frozen_plan_crossing(k=2, h_freeze=0.03, precision=1e-12):
    """Can a plan frozen away from the degeneracy be realized AT it? (No.)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dom0 = Polygon(pentagon(h_freeze), bc='dir')
        plan = BP.plan_basis(dom0, weyl_est(6, dom0), target=precision)
    out = []
    for h in (h_freeze, 1e-3, 1e-5, 1e-8, 0.0, -1e-8, -1e-3):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dom = Polygon(pentagon(h), bc='dir')
                solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom),
                                                    basis=BP.realize(plan, dom), prec=precision)
                lams = np.asarray(Eigenproblem(dom, eval_solver=solver,
                                               precision=precision).solve(k + 2), dtype=float)
            out.append(dict(h=h, n_corners=len(dom.corners), F=float(lams[k]/lams[0])))
        except Exception as exc:
            out.append(dict(h=h, error=f'{type(exc).__name__}: {exc}'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=2, help='0-based; 2 is lam_3/lam_1')
    ap.add_argument('--precision', type=float, default=1e-12)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    exact = 35/11 if args.k == 2 else None
    if exact:
        print(f'exact quadrilateral value: 35/11 = {exact:.12f}\n')
    rows = sweep(args.k, args.precision)

    print(f'{"h":>10} {"corners":>7} {"ncol":>5} {"sigma":>9} {"lam1":>14} {"F":>15}')
    for r in rows:
        if 'error' in r:
            print(f'{r["h"]:10.1e}  {r["error"][:70]}')
            continue
        print(f'{r["h"]:10.1e} {r["n_corners"]:>7} {r["n_cols"]:>5} {r["sigma"]:9.1e} '
              f'{r["lam1"]:14.9f} {r["F"]:15.12f}')

    left, right = one_sided_slopes(rows)
    print(f'\none-sided dF/dh at h=0:  left {left:+.6f}   right {right:+.6f}')
    print('opposite signs => a concave kink; the maximum is AT the degeneracy'
          if left is not None and right is not None and left > 0 > right else
          'NOT a kink maximum here')

    print(f'\nfrozen plan built at h = +0.03, realized across the degeneracy:')
    for r in frozen_plan_crossing(args.k, precision=args.precision):
        if 'error' in r:
            print(f'  h={r["h"]:9.1e}  {r["error"][:70]}')
        else:
            print(f'  h={r["h"]:9.1e}  corners={r["n_corners"]}  F={r["F"]:.12f}')

    if args.json:
        json.dump(dict(k=args.k, rows=rows, left=left, right=right), open(args.json, 'w'))
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
