"""How many basis columns does a regular N-gon want, and does the planner give it that?

    uv run python -m benchmarks.ngon_basis_probe [--N 5 8 12 16] [--targets 1e-10 1e-12 1e-14]

WHY THIS IS ON THE CRITICAL PATH FOR N = 12-16. The scope for this work runs to `N = 16`. The
planner's `n_cap = 240` is a rank-saturation ceiling, and for a regular N-gon the Fourier-Bessel
budget grows with the CORNER COUNT -- one block per corner -- so it crosses the cap early and
stays across it:

    N= 5  n_total = 145   capped = False
    N= 8  n_total = 256   capped = True
    N=12  n_total = 396   capped = True
    N=16  n_total = 560   capped = True

**Every one of those `capped = True` is inoperative, and this is the durable finding here.**
`basis_plan._apply_cap` thins SOURCES to fit, and refuses when the FB budget alone already exceeds
the cap -- and its refusal branch returns the arcs untouched. The sources are additionally already
at `min_src_per_arc`, so there is nothing left to thin even in principle. Sweeping `n_cap` over
{150, 200, 240, 300, 400, 600} produces a BYTE-IDENTICAL plan at every N >= 8: same `n_total`,
same digits to every figure. `plan.capped` reads `True` at 240 and `False` at 400 for the same
plan. Exactly one basis size is reachable at each N, and `target` is the only lever the planner
exposes.

READ THIS BEFORE TRUSTING ANY NUMBER THIS PRODUCES. The first version of this probe did not pass
`prec` to `MPSEigensolver.from_domain`, and every accuracy number it produced was wrong -- it
reported ~8.2 digits at N = 8, 12 and 16, flat against every lever, which read convincingly as a
basis ceiling for the whole N range in scope. It was the eigenvalue SEARCH tolerance, not the
basis. On the unit square, whose spectrum is `pi^2(m^2+n^2)` exactly, the SAME 108-column basis
gives

    default prec :  check_precision 8.84 digits,  true error 4.6e-10  (9.34 digits)
    prec = 1e-13 :  check_precision 13.28 digits, true error 3.1e-14  (13.51 digits)

`douse.SpectrumEvaluator` uses `prec=1e-13`, so that is the operating point any statement about
`douse` has to be made at, and it is the default here. The retracted reading was not a small
error: it turned "N = 8-16 resolves comfortably" into "the N range in scope is out of reach".

WHAT IT SAYS, at `prec=1e-13`, `k=4`, `lam_max = weyl_est(8, dom)`:

    N    target=1e-10        1e-12          1e-14
    5    10.71 (n=120)   11.52 (n=135)  11.14 (n=145)
    8    11.67 (n=200)   11.27 (n=232)  10.97 (n=256)
    12   10.34 (n=312)   10.34 (n=360)  10.67 (n=396)
    16   10.41 (n=432)   10.00 (n=496)  10.31 (n=560)

**Every cell clears 10 digits.** `N = 8-16` is comfortably within the solver's reach, so the scope
decision to run to `N = 12-16` is not blocked by the forward solver. `target` is a weak and
NON-MONOTONE lever -- worth at most ~0.7 digits, and more columns helps at N=12 while hurting at
N=8 -- so there is no case here for changing `SpectrumEvaluator`'s `target=1e-14`.

WHAT IT MEASURES. Achieved digits against `target` (and against `n_cap`, to demonstrate the
inertness above) for the regular N-gon of perimeter 4 -- the shape every one of these runs starts
at, and the conjectured answer for `max lam_2/lam_1`. Resumable ledger in
`run/ngon_basis/probe.jsonl`, keyed on `(N, k, n_cap, window, target, prec)`.
"""
import argparse
import json
import os
import time
import traceback
import warnings
from dataclasses import replace

import numpy as np

from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse import VertexPolygon, param_vector

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ngon_basis')
LEDGER = os.path.join(OUT, 'probe.jsonl')


def regular_domain(N, perim=PERIM):
    fam = VertexPolygon(N, perim=perim)
    v = perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)
    return fam(param_vector(v, perim))


def probe(N, k, n_cap, window, target=1e-14, prec=1e-13):
    rec = dict(N=N, k=k, n_cap=n_cap, window=window, target=float(target), prec=float(prec))
    t0 = time.perf_counter()
    try:
        dom = regular_domain(N)
        lam_max = weyl_est(window, dom)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = replace(BP.PlanConfig(), n_cap=n_cap)
            plan = BP.plan_basis(dom, lam_max, target=target, cfg=cfg)
            # `prec` IS NOT OPTIONAL IN A MEASUREMENT LIKE THIS, and leaving it out invalidated
            # the first version of this probe entirely. It reported ~8.2 digits at N = 8, 12, 16
            # and flat against every basis-size lever, which read as a basis ceiling. It was the
            # eigenvalue SEARCH tolerance: on the unit square, the same 108-column basis gives
            # 8.84 digits at the default `prec` and 13.28 at `prec=1e-13`, against an exactly
            # known spectrum (true error 4.6e-10 against 3.1e-14). `SpectrumEvaluator` uses
            # `prec=1e-13`, so that is the operating point any statement about `douse` must use.
            solver = MPSEigensolver.from_domain(dom, lam_max=lam_max,
                                                basis=BP.realize(plan, dom), prec=prec)
            evp = Eigenproblem(dom, eval_solver=solver, precision=prec)
            eigs = np.asarray(evp.solve(k), dtype=float)
            chk = evp.check_precision(eigs)
        rec.update(ok=True, n_total=int(plan.n_total),
                   n_fb=int(sum(c.M for c in plan.corners)),
                   n_src=int(sum(a.n_src for a in plan.arcs)),
                   capped=bool(plan.capped), digits=float(chk['digits']),
                   lams=[float(t) for t in eigs])
    except Exception as exc:
        rec.update(ok=False, error=f'{type(exc).__name__}: {exc}',
                   trace=traceback.format_exc()[-400:])
    rec['seconds'] = round(time.perf_counter() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='*', default=[8, 12, 16])
    ap.add_argument('--caps', type=int, nargs='*', default=[150, 200, 240, 300, 400, 600])
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--prec', type=float, default=1e-13,
                    help='the eigenvalue search tolerance. SpectrumEvaluator uses 1e-13; the '
                         'lappy default is far looser and DOMINATES the achieved accuracy.')
    ap.add_argument('--targets', type=float, nargs='*', default=[1e-14],
                    help='the planner target. For a regular N-gon at N >= 8 this is the ONLY '
                         'lever: n_cap is inert there because the Fourier-Bessel budget alone '
                         'exceeds every cap and the sources are already at their per-arc floor.')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                r = json.loads(line)
                done.add((r['N'], r['k'], r['n_cap'], r['window'],
                      float(r.get('target', 1e-14)), float(r.get('prec', -1.0))))
            except (ValueError, KeyError):
                pass

    for N in args.N:
        for tgt in args.targets:
            for cap in args.caps:
                if (N, args.k, cap, args.window, float(tgt), float(args.prec)) in done:
                    continue
                rec = probe(N, args.k, cap, args.window, tgt, args.prec)
                with open(LEDGER, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                if rec['ok']:
                    print(f"  N={N:2d} target={tgt:.0e} cap={cap:4d}: "
                          f"n_total={rec['n_total']:4d} "
                          f"(FB {rec['n_fb']:3d} + src {rec['n_src']:3d}) "
                          f"capped={rec['capped']!s:5s} digits={rec['digits']:5.2f}  "
                          f"{rec['seconds']:6.1f}s", flush=True)
                else:
                    print(f"  N={N:2d} target={tgt:.0e} cap={cap:4d}: {rec['error'][:80]}",
                          flush=True)

    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    rows = [r for r in rows if r.get('ok') and r['k'] == args.k and r['window'] == args.window
            and float(r.get('prec', -1.0)) == float(args.prec)]
    print(f'\nregular N-gon, k={args.k}, window={args.window}, prec={args.prec:.0e} '
          f'-- digits (n_total):')
    tgts = sorted({r.get('target', 1e-14) for r in rows}, reverse=True)
    print('  N   ' + ''.join(f'{t:>20.0e}' for t in tgts))
    for N in sorted({r['N'] for r in rows}):
        cells = []
        for t in tgts:
            m = [r for r in rows if r['N'] == N and r.get('target', 1e-14) == t]
            if m:
                best = max(m, key=lambda r: r['digits'])
                spread = len({r['n_total'] for r in m})
                cells.append(f"{best['digits']:5.2f} (n={best['n_total']:3d}{'*' if spread > 1 else ''})")
            else:
                cells.append('')
        print(f'  {N:2d}  ' + ''.join(f'{c:>20s}' for c in cells))
    print("  (* = n_cap changed the basis size at that target; no star means n_cap was inert)")


if __name__ == '__main__':
    main()
