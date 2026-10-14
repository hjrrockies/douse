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

from douse import VertexPolygon, ledger, spectral_guard

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
    from douse import runs

    N, k = row['N'], row['k']
    ev = runs.make_evaluator(N, k, n_eigs=k + 1 + n_extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, _ = ev.eigenvalues(np.array(row['x'], dtype=float))
    lams = np.sort(np.asarray(lams, dtype=float))
    ratio = float(lams[k]/lams[0])
    ok = abs(ratio - row['fun']) <= 1e-6*max(1.0, abs(row['fun']))
    return ratio, lams, ok


PERIM = 4.0


def load(path):
    """The rows of one ledger, DEDUPED BY START, keeping the last attempt at each.

    `ratio_multistart` resumes by start index and appends, and `run_grid` retries a CRASHED start
    on resume -- so a cell can hold more rows than it has starts. `N6_k1` holds 9 for 8 (start 1
    crashed twice) and this function used to report `n_total = 9`, which makes every "certified of
    total" denominator in the table wrong for that cell. No VALUE was ever affected; the filters
    run per row and a duplicate crash row certifies nothing.

    Keeping the LAST is the resume semantics: a retry exists because the earlier attempt failed,
    and the guards a row ran under are recorded on the row, so the newest is the one written by
    the most recent code. Rows with no `start` field (none on disk today) are kept as they are
    rather than collapsed onto each other.
    """
    return ledger.read_rows(path)              # normalized: every row carries both name sets


def row_deficit(r):
    """`worst_weyl_deficit` for a row, RECOMPUTED from `x` and `lams` when it is absent.

    Rows written before the guard existed have no such field, and they are exactly the rows most
    worth auditing. `x` and `lams` are stored, so the statistic is recoverable for every row ever
    written -- an audit of the 81 rows on disk found 3 with a deficit above threshold, none of
    which had certified.

    BUT THE RECOMPUTATION IS THE WEAKER STATISTIC AND THE AUDIT INHERITS THAT. The recorded field
    is the worst deficit over EVERY ITERATE of a run; `x` and `lams` describe the endpoint alone,
    so what comes back here is a lower bound on it. Measured over the 9 rows that carry both
    (2026-08-26): N=6 `lam_3/lam_1` start 3 recorded 8.761 and recomputes to 0.186, which is
    below `DEFICIT_TOL` -- the fallback PASSES a run the live guard failed by a factor of 17.
    Also 5.103 -> 1.284, 6.711 -> 1.022, 2.008 -> 1.328. 71 of the 80 rows on disk are audited
    this way, so "the audit found 3" is a floor, not a count. Pinned in
    `tests/test_multistart_summary.py`.
    """
    d = ledger.normalize(r).get('worst_deficit')
    if d is not None and np.isfinite(d):
        return float(d)
    try:
        dom = VertexPolygon(r['N'], perim=PERIM)(np.asarray(r['x'], dtype=float))
        return float(spectral_guard.weyl_deficit(dom, r['lams']))
    except Exception:                                                 # noqa: BLE001
        return float('nan')


def _fingerprint(r):
    """Sorted edges and angles: a congruence invariant, since congruent polygons are isospectral.

    `None` when the row does not carry them -- rows written before those fields existed, and the
    `pre_plateau_fix` archive. A row without a shape simply does not take part in shape merging;
    it is keyed on `F` alone rather than crashing the summary that reads it.
    """
    if r.get('edges') is None or r.get('angles') is None:
        return None
    return np.concatenate([np.asarray(r['edges']), np.asarray(r['angles'])/180.0])


def find_basins(good, agree_rtol=1e-6, shape_merge_tol=1e-2):
    """Distinct certified optima in a cell, keyed on the OBJECTIVE VALUE, best first.

    KEYED ON `F`, NOT ON THE SHAPE, and the measurement says why. Grouping by a congruence
    fingerprint at a flat tolerance split N=6 `lam_2/lam_1` into two "basins" whose values agree
    to 2.6e-07: three certified runs at the regular hexagon, one of which stopped less sharply
    (resolution 9.2e-05) and so sits 0.28 degrees away in angle, past a 0.18-degree tolerance. A
    resolution-scaled tolerance was the obvious repair and is not defensible either -- the split
    pair sits at 18 times its resolution and the merged pair at 2.2 times, a factor of 8 apart on
    three data points, which is a guess with extra steps.

    `F` is the quantity every cell has 145 measurements of, and `agree_rtol` is already
    calibrated on it. The fingerprint keeps one job: flagging rows that agree in `F` but are NOT
    the same shape, which is the genuine collision and is reported rather than merged.
    """
    basins = []
    for r in sorted(good, key=lambda r: -r['fun']):
        for b in basins:
            if abs(r['fun'] - b['fun']) <= agree_rtol*abs(b['fun']):
                b['rows'].append(r)
                break
        else:
            basins.append(dict(fun=r['fun'], rows=[r]))
    # NEITHER KEY WORKS ALONE, and both failures were measured. Keying on the SHAPE split N=6
    # `lam_2/lam_1` into two, on values agreeing to 2.6e-07, because one run stopped less sharply
    # and sits 0.28 degrees away. Keying on `F` alone splits N=3 `lam_6/lam_1` into two, on the
    # same triangle to 0.04 degrees, because the less-sharp run differs in the fifth figure.
    #
    # So: key on `F`, then merge what the SHAPE says is one optimum. Measured over every
    # multi-basin pair in the grid, the two classes separate by two orders of magnitude in shape
    # distance -- 4.1e-04 for the one pair that is the same optimum, against 5.0e-02 to 8.0e-01
    # for the five that are genuinely distinct. `shape_merge_tol = 1e-2` sits 24x above the first
    # and 5x below the nearest of the second.
    merged = []
    for b in basins:
        fb = _fingerprint(b['rows'][0])
        for m in merged:
            fm = _fingerprint(m['rows'][0])
            if fb is None or fm is None or len(fb) != len(fm):
                continue
            if float(np.max(np.abs(fb - fm))) < shape_merge_tol:
                m['rows'].extend(b['rows'])
                break
        else:
            merged.append(b)
    basins = merged
    for b in basins:
        b['fun'] = max(r['fun'] for r in b['rows'])
        fps = [f for f in (_fingerprint(r) for r in b['rows']) if f is not None]
        b['shape_spread'] = float(max((np.max(np.abs(f - fps[0])) for f in fps), default=0.0))
        b['best'] = max(b['rows'], key=lambda r: r['fun'])
        b['spread'] = (float(np.ptp([r['fun'] for r in b['rows']])) if len(b['rows']) > 1
                       else float('nan'))
    return basins


def summarize(rows, sigma_max, res_max, agree_rtol=1e-6, near_rtol=1e-5):
    """Best certified value in a cell, and how many starts agree with it."""
    rows = [ledger.normalize(r) for r in rows]
    crashed = [r for r in rows if not r.get('ok', True)]
    ok = [r for r in rows if r.get('ok', True)]
    # A ROW WITHOUT `worst_bound` IS UNCERTIFIED, not invisible: `nan` fails both comparisons
    # below, and every chladni row used to fall out of every count here (2026-10-14). On a v1
    # row the statistic is lappy's tension under the same name (`douse.ledger`).
    sig = lambda r: (float('nan') if r.get('worst_bound') is None                 # noqa: E731
                     else float(r['worst_bound']))
    uncertified = [r for r in ok if not np.isfinite(sig(r))]
    drifted = [r for r in ok if sig(r) > sigma_max]
    clean = [r for r in ok if sig(r) <= sigma_max]
    # A NON-CONSECUTIVE TRACKED SET IS NOT A RESULT, and `worst_sigma` above cannot see one:
    # the 2.6203 run had sigma = 1e-09. Dropped here rather than merely reported, because a
    # value promoted out of this function is quoted.
    skipped = [r for r in clean if row_deficit(r) > spectral_guard.DEFICIT_TOL]
    clean = [r for r in clean if not (row_deficit(r) > spectral_guard.DEFICIT_TOL)]
    # A CERTIFICATE IS THE STOP, NOT A THRESHOLD ON THE RESOLUTION -- section 9's lesson, which
    # this line was repeating one layer up. `gain_tol` was replaced by plateau detection because a
    # fixed threshold "certified according to where each shape's noise floor happened to fall";
    # `res_max = 1e-4` was then applied to the PLATEAU VALUE and did the same thing. Measured over
    # the 20 rows it was rejecting: **17 agree with their cell's best to 1e-6**, most to 1e-8, and
    # every one of them stopped on the plateau certificate. Of the other three, two sit 2.8e-06
    # and 3.3e-06 out -- the same optimum, less sharply -- and one is a different basin. None was
    # junk, and N=3 `lam_5/lam_1` was reported NONE CERTIFIED while six starts agreed on it.
    #
    # The upper bound that DOES belong is already in the optimizer, where the measurement is:
    # `maximize_ratio` only runs the plateau test when criticality is under `plateau_max = 1e-3`,
    # so a run that plateaued absurdly high never certifies in the first place. Accuracy is the
    # other guards' job (sigma, the Weyl deficit, the spurious-value bar); resolution is an
    # OPTIMALITY measure, and gating accuracy on it is what conflates the two.
    #
    # `res_max` survives as a reported sharpness, `n_sharp`, not as a gate.
    good = [r for r in clean if r.get('success')]
    n_sharp = sum(1 for r in good if np.isfinite(r.get('resolution', np.nan))
                  and r['resolution'] <= res_max)
    if not good:
        return dict(best=None, n_total=len(rows), n_crashed=len(crashed),
                    schema_mix=ledger.schema_mix(rows),
                    n_skipped_mode=len(skipped), n_sharp=0, n_uncertified=len(uncertified),
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
    basins = find_basins(good, agree_rtol)
    return dict(best=best, basins=basins, n_total=len(rows), n_crashed=len(crashed), n_clean=len(clean),
                n_skipped_mode=len(skipped), n_sharp=n_sharp, n_uncertified=len(uncertified),
                schema_mix=ledger.schema_mix(rows),
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
        # A CONVEX CELL IS A DIFFERENT QUESTION, not more rows for the same one. `--convex`
        # writes `N<N>_k<k>_convex.jsonl`, which this glob picks up, and pooling it with the
        # unconstrained cell of the same `(N, k)` would silently mix a constrained optimum into
        # an unconstrained answer. Keyed and labelled separately instead.
        N, k = rows[0]['N'], rows[0]['k']
        conv = bool(rows[0].get('convex', False)) or path.endswith('_convex.jsonl')
        cells.append((N, k, conv, summarize(rows, args.sigma_max, args.res_max)))
    cells.sort(key=lambda c: (c[0], c[1], c[2]))

    print(f'{len(cells)} cell(s) from {args.dir}\n')
    print('   N   k  family   max lam_k/lam_1     spread   resolution  agree near cert/tot'
          ' sharp  oracle (N=4)')
    print('  ' + '-'*101)
    for N, k, conv, s in cells:
        fam = 'convex' if conv else 'all   '
        if s['best'] is None:
            got = (f"{s['best_uncertified']:.10f}" if s['best_uncertified'] is not None else '-')
            print(f"  {N:2d}  {k+1:2d}  {fam}  {got:>16s}   NONE CERTIFIED    -   "
                  f"0/{s['n_total']-s['n_crashed']-s['n_drifted']-s['n_uncertified']}/"
                  f"{s['n_total']}")
            continue
        b = s['best']
        oracle = ''
        if N == 4 and best_rectangle is not None:
            _, bound = best_rectangle(k, 4.0)
            delta = b['fun'] - bound
            oracle = (f'  {bound:.6f} ({delta:+.2e})'
                      + ('  BELOW -- not converged' if delta < -1e-7 else ''))
        print(f"  {N:2d}  {k+1:2d}  {fam}  {b['fun']:16.10f}  {s['spread']:8.1e}  "
              f"{b['resolution']:10.2e}  {s['n_agree']:2d}/{s['n_certified']:<2d} "
              f"{s['n_near']:2d}/{s['n_clean']:<2d} {s['n_certified']}/{s['n_total']}"
              f" {s['n_sharp']:2d}/{s['n_certified']:<2d}{oracle}")

    print('\n  shapes of the best certified point in each cell:')
    for N, k, conv, s in cells:
        if s['best'] is None:
            continue
        b = s['best']
        print(f"  N={N:2d} k={k+1:2d}{' convex' if conv else ''}: angles {np.array2string(np.asarray(b['angles']), precision=2)}")
        print(f"           edges  {np.array2string(np.asarray(b['edges']), precision=4)}")

    # MORE THAN ONE CERTIFIED OPTIMUM IS A RESULT, not a nuisance, and reporting only the best
    # threw it away. N=4 `lam_4/lam_1` splits 4-4 between an isosceles trapezoid and the sqrt(5)
    # rectangle -- both genuine local maxima, and the ~1000:1 curvature ratio in the handoff
    # predicts the split. Occupancy is evidence about relative basin size and is printed with it.
    multi = [(N, k, c, s) for N, k, c, s in cells if len(s.get('basins') or []) > 1]
    if multi:
        print('\n  cells with more than one certified optimum:')
        for N, k, conv, s in multi:
            print(f"  N={N:2d} k={k+1:2d}{' convex' if conv else ''}:")
            for b in s['basins']:
                ang = np.array2string(np.asarray(b['best']['angles']), precision=2)
                print(f"      {b['fun']:15.10f}  {len(b['rows'])} start(s)  angles {ang}")
            # NEARLY EQUAL F AT GENUINELY DIFFERENT SHAPES is the one case worth a second look --
            # two distinct maximizers with the same ratio, which is either a symmetry of the
            # problem or something more interesting. Reported, never merged.
            for x, y in zip(s['basins'][:-1], s['basins'][1:]):
                fx, fy = _fingerprint(x['best']), _fingerprint(y['best'])
                if abs(x['fun'] - y['fun']) <= 1e-5*abs(x['fun']) and fx is not None \
                        and fy is not None and len(fx) == len(fy):
                    d = float(np.max(np.abs(fx - fy)))
                    print(f"      ^ these differ in F by only "
                          f"{abs(x['fun']-y['fun'])/abs(x['fun']):.1e} at a shape distance of "
                          f"{d:.1e} -- different shapes, nearly equal ratio")

    # PPW is a THEOREM, so a cell above it is wrong no matter what it certified.
    viol = [(N, k, s) for N, k, _c, s in cells
            if k == 1 and s['best'] is not None and s['best']['fun'] > PPW + 1e-9]
    for N, k, s in viol:
        print(f"\n  *** N={N} k=2 reports {s['best']['fun']:.10f}, ABOVE the "
              f"Payne-Polya-Weinberger bound {PPW:.7f}. No planar domain can do this; the "
              f"spectrum is wrong (usually a MISSED eigenvalue). ***")

    if args.verify:
        print('\n  re-solving each best shape on a fresh plan:')
        for N, k, _conv, s in cells:
            if s['best'] is None:
                continue
            ratio, lams, ok = revalidate(s['best'])
            mark = 'ok' if ok else f"MISMATCH -- fresh solve gives {ratio:.10f}"
            print(f"  N={N:2d} k={k+1:2d}: recorded {s['best']['fun']:.10f}   {mark}")
            if not ok:
                print(f"       fresh lams {np.array2string(lams[:k+2], precision=6)}")

    total = lambda key: sum(s[key] for *_, s in cells)                 # noqa: E731
    tot, cert = total('n_total'), total('n_certified')
    crash, drift, missed = total('n_crashed'), total('n_drifted'), total('n_skipped_mode')
    unc = total('n_uncertified')
    mix = {}
    for *_, c in cells:
        for k2, v2 in c.get('schema_mix', {}).items():
            mix[k2] = mix.get(k2, 0) + v2
    print('\n  rows by ledger schema: ' + ', '.join(f'v{k2}: {v2}' for k2, v2 in sorted(mix.items())))
    print(f'\n  {cert} certified of {tot} run(s); {drift} excluded on worst_sigma, '
          f'{missed} on a missed mode, {crash} crashed, {unc} without a health statistic')
    print('  agree = certified starts that found the best value; CERTIFIED means the run')
    print('          stopped on a first-order certificate (gain_tol or the plateau test).')
    print(f'  sharp = of those, how many reached resolution <= {args.res_max:.0e}. A REPORTED')
    print('          quality, not a gate: the plateau height is set by the shape\'s own noise')
    print('          floor, so thresholding it certifies by geometry rather than by optimality.')
    print('  near  = ALL clean starts within 1e-5 of it -- the same shape found less sharply,')
    print('          which is the globality evidence; resolution is a separate question.')
    print('  spread = how far apart the agreeing certified starts ended in F (the uncertainty).')
    print('  resolution = first-order optimality residual; F error goes as its SQUARE, so these')
    print('  differ by orders of magnitude and the resolution is NOT an error bar on F.')


if __name__ == '__main__':
    main()
