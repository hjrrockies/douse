"""Re-solve a REFUSED shape off-line: plan on one shape, solve on the other, and see what broke.

    uv run python -m benchmarks.replay_refusal --cell N6_k2 [--start 3] [--index 0]

A `UnresolvedSpectrum` refusal says the solved set is not `lam_1 .. lam_K` at that shape. The
count alone is not actionable -- the question it raises is a `lappy` question, and `lappy` needs
the VERTICES. Ledger rows now carry up to four refusals each, with both ends of the pair: the `p`
that failed and the `plan_at` its frozen basis was planned on.

WHAT THIS ANSWERS THAT THE SWEEP COULD NOT. `lappy`'s `benchmarks/envelope/reentrant_sweep.py`
walks a notched hexagon from 150 to 292 degrees and finds no missing mode anywhere, at any
staleness it was asked for -- so the shapes that defeat a full solve have something that family
does not, and guessing at what is exactly the loop this closes. Run this on a real refusal and
the three rows it prints -- fresh plan at the failing shape, the frozen plan it actually used,
and the plan's own shape -- separate "this geometry is out of reach" from "this PLAN was".
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np

from douse import VertexPolygon, spectral_guard
from douse.evaluate import SpectrumEvaluator

PERIM = 4.0
LEDGERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_multistart')


def solve_at(fam, k, p, plan_at=None):
    """`(lams, sigma, deficit)` for `p`, on a basis planned at `plan_at` (default: at `p`)."""
    ev = SpectrumEvaluator(fam, n_eigs=k + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*k + 4), replan_tol=None,
                           plan_at=(p if plan_at is None else plan_at), enforce_weyl=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, _ = ev.eigenvalues(np.asarray(p, dtype=float))
        sigma = ev.sigma_at(p, lams[0])
    dom = fam(np.asarray(p, dtype=float))
    return lams, sigma, float(spectral_guard.weyl_deficit(dom, lams))


def report(row, index):
    ref = row['refusals'][index]
    fam = VertexPolygon(row['N'], perim=PERIM)
    p, plan_at = np.array(ref['p'], float), ref.get('plan_at')
    print(f"\nN={row['N']} k={row['k']} start={row['start']} refusal {index}: "
          f"recorded deficit {ref['deficit']:+.2f}, from_full_solve={ref.get('from_full_solve')}")
    for name, q, plan in (('the refused shape, FRESH plan', p, None),
                          ('the refused shape, the FROZEN plan it used', p, plan_at),
                          ("the plan's own shape, fresh", plan_at, None)):
        if q is None:
            continue
        lams, sigma, deficit = solve_at(fam, row['k'], q, plan)
        v = fam.vertices(np.asarray(q, float))
        from douse import degeneracy as dg
        a = np.degrees(dg.interior_angles(v))
        e = np.abs(v - np.roll(v, 1))
        flag = '  MODE MISSING' if deficit > spectral_guard.DEFICIT_TOL else ''
        print(f"  {name:44s} sigma {sigma:8.1e}  deficit {deficit:+6.2f}{flag}")
        print(f"  {'':44s} max_ang {a.max():6.1f}  edge ratio {e.min()/e.max():.3f}  "
              f"lam {np.array2string(lams[:4], precision=4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, help='ledger stem, e.g. N6_k2')
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--index', type=int, default=None, help='which refusal in the row')
    ap.add_argument('--dir', default=LEDGERS)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, f'{args.cell}.jsonl')))
    if not paths:
        print(f'no ledger {args.cell} in {args.dir}')
        return
    rows = [json.loads(l) for l in open(paths[0]) if l.strip()]
    rows = [r for r in rows if r.get('refusals')
            and (args.start is None or r['start'] == args.start)]
    if not rows:
        print('no rows with recorded refusals -- rows written before the enforcement have none')
        return
    for r in rows:
        idx = ([args.index] if args.index is not None else range(len(r['refusals'])))
        for i in idx:
            report(r, i)


if __name__ == '__main__':
    main()
