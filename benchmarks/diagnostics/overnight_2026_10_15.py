"""Overnight 2026-10-15: converged runs on both warm paths, and the cadence at 20.

Three N=6, k=2 (lam_3/lam_1) seeds, each to convergence (max_iter 400), under three settings:
`windowed` (today's default: step-controlled per-level windows), `prefix` (the warm solve
searching the full prefix, as before 2026-10-14; forced by giving the optimizer a solved shape
that cannot predict moves), and `windowed, cadence 20` (chladni's cold-solve backstop every 20
warm solves instead of 10). Rows (schema 2, with the per-solve trace) go to
`benchmarks/run/diagnostics/overnight_2026_10_15.jsonl`; `--report` tabulates them.

The questions: do the two warm paths agree at the optimum to 1e-9 (the plan's open check), how
much wall each saves, and what the cadence buys.

Settings ending `_ss` (added 2026-10-16) are the same three paths rerun on the scalar-secant
curvature model, the default since douse `eeae9e1`; the unsuffixed rows were on SR1 and stay
for comparison. The path is read from the prefix of the name (`prefix*`, `cadence20*`, else
windowed). Baseline commands, three at a time:

    uv run python benchmarks/diagnostics/overnight_2026_10_15.py --settings windowed_ss --seeds 0 1 2
    uv run python benchmarks/diagnostics/overnight_2026_10_15.py --settings prefix_ss --seeds 0 1 2
    uv run python benchmarks/diagnostics/overnight_2026_10_15.py --settings cadence20_ss --seeds 0 1 2

(each `--seeds` list runs in sequence; launch one process per seed to run three at once).
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
    if setting.startswith('cadence20'):
        EP.WARM_COLD_EVERY = 20
    evaluator = _PrefixEvaluator(ev) if setting.startswith('prefix') else ev
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


def _stop(r):
    """'budget' when the run used every iterate, else 'rule' (it stopped on its own test)."""
    return 'budget' if r['nit'] >= r.get('max_iter', 10**9) else 'rule'


def report():
    rows = ledger.read_rows(OUT, dedupe=False)
    by = {}
    for r in rows:
        by.setdefault(r['start'], {})[r['setting']] = r
    print(f"{'seed':>4} {'setting':>12} {'F':>14} {'nit':>4} {'s':>6} {'warm':>5} {'cold':>5} "
          f"{'replan':>6} {'evals':>7} {'ev/it':>6} {'stop':>6}  message")
    for seed in sorted(by):
        for setting, r in sorted(by[seed].items()):
            tr = r.get('trace', [])
            kinds = {}
            for t in tr:
                kinds[t['kind']] = kinds.get(t['kind'], 0) + 1
            ev = r.get('n_tension_evals', 0)
            print(f"{seed:4d} {setting:>12} {r['fun']:14.10f} {r['nit']:4d} {r['seconds']:6.0f} "
                  f"{kinds.get('warm', 0):5d} {kinds.get('cold', 0):5d} "
                  f"{kinds.get('replan', 0):6d} {ev:7d} {ev / max(r['nit'], 1):6.1f} "
                  f"{_stop(r):>6}  {r['message'][:40]}")
        # prefix against the matching windowed setting: SR1 rows against the latest SR1
        # windowed run (windowed2 is the fixed step cap, 271e740), `_ss` rows against `_ss`
        s = by[seed]
        pairs = []
        wins = sorted(k for k in s if k.startswith('windowed') and not k.endswith('_ss'))
        if 'prefix' in s and wins:
            pairs.append(('prefix', wins[-1]))
        if 'prefix_ss' in s and 'windowed_ss' in s:
            pairs.append(('prefix_ss', 'windowed_ss'))
        if 'cadence20_ss' in s and 'windowed_ss' in s:
            pairs.append(('cadence20_ss', 'windowed_ss'))
        for a, b in pairs:
            print(f"     {a} vs {b}: |dF| = {abs(s[a]['fun'] - s[b]['fun']):.2e}  "
                  f"wall {s[a]['seconds']:.0f}s vs {s[b]['seconds']:.0f}s")


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
