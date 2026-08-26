"""One table for the whole sweep: what `max lam_k/lam_1` is, and how well it is known.

    uv run python -m benchmarks.summarize_grid [--sigma-max 1e-6] [--res-max 1e-4]

Reads every ledger in `run/ratio_multistart/` and reports, per `(N, k)` cell: the best certified
value, the resolution it is certified to, how many independent starts agree with it, and the
shape. This is the form the results want to be quoted in, and it is deliberately NOT "the largest
`F` anyone saw" -- that number is available from any single run and means nothing on its own.

THREE FILTERS, IN THIS ORDER, AND EACH ONE HAS CAUGHT SOMETHING REAL:

  1. `worst_sigma` -- did the frozen basis still resolve the shape? A run that drifted is not a
     result; ignoring this is what produced the retracted 3.0717 for a problem whose answer is
     above 3.18.
  2. `success` and `resolution` -- did the run stop on a certificate, and to what resolution?
     Criticality plateaus at the shape-derivative noise floor rather than reaching zero, so
     "first-order optimal" without a resolution attached is the claim the plateau stop exists to
     stop making. Measured separation: junk shapes 6.2e+04, unconverged runs 2.1e-03, converged
     runs <= 5.5e-06 -- three orders, so this partitions a ledger on its own.
  3. AGREEMENT ACROSS STARTS. A certificate is local. `n_agree` counts the starts that reached
     the same value; a cell where 8 of 8 independent random starts certify the same shape is the
     strongest statement this machinery can make, and a cell where 2 of 8 do is a warning that
     the basin structure is richer than the run explored.

At N=4 the exact rectangle oracle (`benchmarks/rectangle_exact.py`) is a rigorous LOWER bound, so
it is printed alongside: `F` below it means the cell has NOT converged whatever it certified,
and `F` above it means the optimal quadrilateral is not a rectangle, which is a result.

`resolution` IS NOT AN ERROR BAR ON `F`, and reading it as one badly understates these results.
It is the first-order optimality residual -- roughly `|grad F|` at the stopping point. Near a
maximum `F` is flat, so `|F - F*| ~ crit^2 / (2 |H|)`: the error in `F` goes as the SQUARE of the
resolution, up to curvature. Measured in the N=4 `lam_3/lam_1` cell: every run reports a
resolution of 2-5e-05, and the eight of them agree with each other in `F` to 6.3e-09 -- four
orders tighter than a naive reading of the resolution, and the same order as the `crit^2/2 ~
2e-10` the quadratic argument gives (which is a scaling argument, not a bound: the curvature
factor and the tail of the start distribution both move it, and here they move it by ~30x).

So the table prints both. `resolution` says how close to critical the point is; `spread` says how
far apart the independent starts that certified ended up, and is the empirical statement about how
well the VALUE is pinned. Quote the spread as the uncertainty and the resolution as the
optimality evidence.
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np

PPW = (3.8317059702/2.4048255577)**2      # Payne--Polya--Weinberger: lam_2/lam_1 <= this

try:
    from .rectangle_exact import best_rectangle
except ImportError:                                                   # pragma: no cover
    best_rectangle = None

LEDGERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_multistart')


def revalidate(row, n_extra=3):
    """Re-solve a recorded shape on a FRESH plan and compare. Catches a MISSED eigenvalue.

    THIS IS NOT REDUNDANT WITH `worst_sigma`, and the difference is the whole point. Sigma says how
    well the basis resolves the values it FOUND; it says nothing about whether one was skipped. A
    plan frozen at the starting shape can, at some later shape, miss a mode entirely while
    reporting a perfect tension on every value it does return -- so the run silently optimizes
    `lam_{k+1}/lam_1` while believing it is optimizing `lam_k/lam_1`.

    Measured, N=6 `lam_2/lam_1`, start 2: recorded `[17.082832, 44.762438, 74.090865]` with
    `worst_sigma = 1e-09` and a reported ratio of **2.6203**, which is ABOVE the
    Payne--Polya--Weinberger bound of 2.5387 that no planar domain can exceed. `lam_2 = 41.49929`
    was simply absent. A fresh solve at the same `p` finds it under every `(n_eigs, window)` tried
    and gives 2.4293. `verify_final` did not catch it because it re-solves on the SAME frozen plan.

    Returns `(ratio, lams, ok)` -- `ok` is False when the fresh solve disagrees with the record.
    """
    from douse import SpectrumEvaluator, VertexPolygon

    N, k = row['N'], row['k']
    fam = VertexPolygon(N, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=k + 1 + n_extra, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*k + 6), replan_tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, _ = ev.eigenvalues(np.array(row['x'], dtype=float))
    lams = np.sort(np.asarray(lams, dtype=float))
    ratio = float(lams[k]/lams[0])
    ok = abs(ratio - row['fun']) <= 1e-6*max(1.0, abs(row['fun']))
    return ratio, lams, ok


PERIM = 4.0


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(rows, sigma_max, res_max, agree_rtol=1e-6, near_rtol=1e-5):
    """Best certified value in a cell, and how many starts agree with it."""
    crashed = [r for r in rows if not r.get('ok', True)]
    ok = [r for r in rows if r.get('ok', True)]
    drifted = [r for r in ok if r['worst_sigma'] > sigma_max]
    clean = [r for r in ok if r['worst_sigma'] <= sigma_max]
    good = [r for r in clean
            if r.get('success') and np.isfinite(r.get('resolution', np.nan))
            and r['resolution'] <= res_max]
    if not good:
        return dict(best=None, n_total=len(rows), n_crashed=len(crashed),
                    n_drifted=len(drifted), n_certified=0, n_agree=0, n_near=0,
                    n_clean=len(clean),
                    best_uncertified=(max((r['fun'] for r in clean), default=None)))
    best = max(good, key=lambda r: r['fun'])
    agreeing = [r for r in good if abs(r['fun'] - best['fun']) <= agree_rtol*abs(best['fun'])]
    spread = float(np.ptp([r['fun'] for r in agreeing])) if len(agreeing) > 1 else float('nan')
    # `n_near` COUNTS THE SAME SHAPE FOUND LESS SHARPLY, and without it the table understates the
    # evidence badly. Measured at N=5 `lam_4/lam_1`: five starts certified the same optimum, but
    # four of them at a resolution above `res_max`, so `agree` read 1/1 -- which looks like a
    # single unsupported run when in fact five independent starts landed within 3e-06 of each
    # other. Resolution is about how sharply a point is pinned; agreement is about whether the
    # starts found the SAME point, and conflating them throws away the globality evidence.
    n_near = sum(abs(r['fun'] - best['fun']) <= near_rtol*abs(best['fun']) for r in clean)
    return dict(best=best, n_total=len(rows), n_crashed=len(crashed), n_clean=len(clean),
                n_drifted=len(drifted), n_certified=len(good), n_agree=len(agreeing),
                n_near=n_near, spread=spread, best_uncertified=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma-max', type=float, default=1e-6)
    ap.add_argument('--res-max', type=float, default=1e-4)
    ap.add_argument('--dir', default=LEDGERS)
    ap.add_argument('--verify', action='store_true',
                    help='re-solve each reported best shape on a FRESH plan and flag disagreement '
                         '(catches a missed eigenvalue, which `worst_sigma` cannot). One solve '
                         'per cell.')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, 'N*_k*.jsonl')))
    if not paths:
        print(f'no ledgers in {args.dir}')
        return
    cells = []
    for path in paths:
        rows = load(path)
        if not rows:
            continue
        N, k = rows[0]['N'], rows[0]['k']
        cells.append((N, k, summarize(rows, args.sigma_max, args.res_max)))
    cells.sort(key=lambda c: (c[0], c[1]))

    print(f'{len(cells)} cell(s) from {args.dir}\n')
    print('   N   k   max lam_k/lam_1     spread   resolution  agree near cert/tot  oracle (N=4)')
    print('  ' + '-'*85)
    for N, k, s in cells:
        if s['best'] is None:
            got = (f"{s['best_uncertified']:.10f}" if s['best_uncertified'] is not None else '-')
            print(f"  {N:2d}  {k+1:2d}   {got:>16s}   NONE CERTIFIED    -   "
                  f"0/{s['n_total']-s['n_crashed']-s['n_drifted']}/{s['n_total']}")
            continue
        b = s['best']
        oracle = ''
        if N == 4 and best_rectangle is not None:
            _, bound = best_rectangle(k, 4.0)
            delta = b['fun'] - bound
            oracle = (f'  {bound:.6f} ({delta:+.2e})'
                      + ('  BELOW -- not converged' if delta < -1e-7 else ''))
        print(f"  {N:2d}  {k+1:2d}   {b['fun']:16.10f}  {s['spread']:8.1e}  "
              f"{b['resolution']:10.2e}  {s['n_agree']:2d}/{s['n_certified']:<2d} "
              f"{s['n_near']:2d}/{s['n_clean']:<2d} {s['n_certified']}/{s['n_total']}"
              f"{oracle}")

    print('\n  shapes of the best certified point in each cell:')
    for N, k, s in cells:
        if s['best'] is None:
            continue
        b = s['best']
        print(f"  N={N:2d} k={k+1:2d}: angles {np.array2string(np.asarray(b['angles']), precision=2)}")
        print(f"           edges  {np.array2string(np.asarray(b['edges']), precision=4)}")

    # PPW is a THEOREM, so a cell above it is wrong no matter what it certified.
    viol = [(N, k, s) for N, k, s in cells
            if k == 1 and s['best'] is not None and s['best']['fun'] > PPW + 1e-9]
    for N, k, s in viol:
        print(f"\n  *** N={N} k=2 reports {s['best']['fun']:.10f}, ABOVE the "
              f"Payne-Polya-Weinberger bound {PPW:.7f}. No planar domain can do this; the "
              f"spectrum is wrong (usually a MISSED eigenvalue). ***")

    if args.verify:
        print('\n  re-solving each best shape on a fresh plan:')
        for N, k, s in cells:
            if s['best'] is None:
                continue
            ratio, lams, ok = revalidate(s['best'])
            mark = 'ok' if ok else f"MISMATCH -- fresh solve gives {ratio:.10f}"
            print(f"  N={N:2d} k={k+1:2d}: recorded {s['best']['fun']:.10f}   {mark}")
            if not ok:
                print(f"       fresh lams {np.array2string(lams[:k+2], precision=6)}")

    tot = sum(s['n_total'] for _, _, s in cells)
    cert = sum(s['n_certified'] for _, _, s in cells)
    crash = sum(s['n_crashed'] for _, _, s in cells)
    drift = sum(s['n_drifted'] for _, _, s in cells)
    print(f'\n  {cert} certified of {tot} run(s); {drift} excluded on worst_sigma, {crash} crashed')
    print('  agree = certified starts (resolution <= res_max) that found the best value.')
    print('  near  = ALL clean starts within 1e-5 of it -- the same shape found less sharply,')
    print('          which is the globality evidence; resolution is a separate question.')
    print('  spread = how far apart the agreeing certified starts ended in F (the uncertainty).')
    print('  resolution = first-order optimality residual; F error goes as its SQUARE, so these')
    print('  differ by orders of magnitude and the resolution is NOT an error bar on F.')


if __name__ == '__main__':
    main()
