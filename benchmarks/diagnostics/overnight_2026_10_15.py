"""Overnight 2026-10-15: converged runs on both warm paths, and the cadence at 20.

Three N=6, k=2 (lam_3/lam_1) seeds, each to convergence (max_iter 400), under three settings:
`windowed` (today's default: step-controlled per-level windows), `prefix` (the warm solve
searching the full prefix, as before 2026-10-14; forced by giving the optimizer a solved shape
that cannot predict moves), and `windowed, cadence 20` (chladni's cold-solve backstop every 20
warm solves instead of 10). Rows (schema 2, with the per-solve trace) go to
`benchmarks/run/diagnostics/overnight_2026_10_15.jsonl`; `--report` tabulates them.

The questions: do the two warm paths agree at the optimum to 1e-9 (the plan's open check), how
much wall each saves, and what the cadence buys.
"""
import argparse
import os
import time

import numpy as np

from douse import ledger, runs
from douse.cluster_opt import maximize_ratio
from douse.polygon import param_vector

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run',
                   'diagnostics', 'overnight_2026_10_15.jsonl')


class _NoPrediction:
    """Wrap a solved shape so the optimizer cannot predict moves: the prefix warm path."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in ('predicted_moves',):
            raise AttributeError(name)
        return getattr(self._inner, name)


class _PrefixEvaluator:
    def __init__(self, inner):
        self._inner = inner

    def eigenvalues(self, p, **kw):
        lams, solved = self._inner.eigenvalues(p, **kw)
        return lams, _NoPrediction(solved)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_one(seed, setting, max_iter):
    import chladni.eigenproblem as EP
    N, k = 6, 2
    fam = runs.family_for(N)
    ev = runs.make_evaluator(N, k, fam=fam)
    cadence = EP.WARM_COLD_EVERY
    if setting == 'cadence20':
        EP.WARM_COLD_EVERY = 20
    evaluator = _PrefixEvaluator(ev) if setting == 'prefix' else ev
    t0 = time.perf_counter()
    try:
        res = maximize_ratio(evaluator, param_vector(runs.perturbed(N, 0.08, seed), runs.PERIM),
                             k=k, max_iter=max_iter, constraints=runs.constraints_for(fam))
    finally:
        EP.WARM_COLD_EVERY = cadence
    row = ledger.result_row(res, health=ev.health(), N=N, k=k, start=seed, backend='chladni',
                            prec=ev.prec, seconds=round(time.perf_counter() - t0, 1),
                            setting=setting, max_iter=max_iter)
    ledger.write_row(OUT, row)
    print(f"{setting:10s} seed {seed}: F={row['fun']:.10f} nit={row['nit']} "
          f"{row['seconds']:.0f}s success={row['success']} {row['message'][:50]}", flush=True)
    return row


def report():
    rows = ledger.read_rows(OUT, dedupe=False)
    by = {}
    for r in rows:
        by.setdefault(r['start'], {})[r['setting']] = r
    print(f"{'seed':>4} {'setting':>10} {'F':>14} {'nit':>4} {'s':>6} {'warm':>5} {'cold':>5} "
          f"{'replan':>6} {'evals':>7}  stop")
    for seed in sorted(by):
        for setting, r in sorted(by[seed].items()):
            tr = r.get('trace', [])
            kinds = {}
            for t in tr:
                kinds[t['kind']] = kinds.get(t['kind'], 0) + 1
            print(f"{seed:4d} {setting:>10} {r['fun']:14.10f} {r['nit']:4d} {r['seconds']:6.0f} "
                  f"{kinds.get('warm', 0):5d} {kinds.get('cold', 0):5d} "
                  f"{kinds.get('replan', 0):6d} {r.get('n_tension_evals', 0):7d}  "
                  f"{'ok' if r['success'] else '--'} {r['message'][:40]}")
        if 'prefix' in by[seed] and 'windowed' in by[seed]:
            a, b = by[seed]['prefix']['fun'], by[seed]['windowed']['fun']
            print(f"     prefix vs windowed: |dF| = {abs(a - b):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--settings', nargs='+', default=['windowed', 'prefix', 'cadence20'])
    ap.add_argument('--max-iter', type=int, default=400)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if not args.report:
        done = {(r['start'], r['setting']) for r in ledger.read_rows(OUT, dedupe=False)}
        for setting in args.settings:
            for seed in args.seeds:
                if (seed, setting) not in done:
                    run_one(seed, setting, args.max_iter)
    report()


if __name__ == '__main__':
    main()
