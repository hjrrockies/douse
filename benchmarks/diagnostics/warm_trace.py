"""Where do a k=2 run's solves go?  (2026-10-14; Session 4 step 1 of the follow-up plan)

Runs `maximize_ratio` on chladni for N=6, k=2 (lam_3/lam_1), seeds 0-2, as `chladni_vs_lappy`
does (constrained, 80 iterations), and tabulates the backend's per-solve trace: cold / warm /
plan / replan, the reason a warm solve went cold or a replan was needed, the rung, evaluations
and seconds. The hypothesis under test: most replans are `unmet` on a rung accepted at a margin
barely above 2.

    uv run python -m benchmarks.diagnostics.warm_trace [--seeds 0 1 2] [--max-iter 80]

Rows go to `benchmarks/run/diagnostics/warm_trace.jsonl` (schema 2, with the trace).
"""
import argparse
import collections
import os
import time

import numpy as np

from douse import ledger, runs
from douse.cluster_opt import maximize_ratio
from douse.polygon import param_vector

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run',
                   'diagnostics', 'warm_trace.jsonl')


def run_one(N, k, seed, max_iter):
    fam = runs.family_for(N)
    ev = runs.make_evaluator(N, k, fam=fam)
    cons = runs.constraints_for(fam)
    t0 = time.perf_counter()
    res = maximize_ratio(ev, param_vector(runs.perturbed(N, 0.08, seed), runs.PERIM), k=k,
                         max_iter=max_iter, constraints=cons)
    row = ledger.result_row(res, health=ev.health(), N=N, k=k, start=seed, backend='chladni',
                            prec=ev.prec, seconds=round(time.perf_counter() - t0, 1),
                            max_iter=max_iter)
    ledger.write_row(OUT, row)
    return row


def tabulate(rows):
    for r in rows:
        tr = r.get('trace', [])
        kinds = collections.Counter(t['kind'] for t in tr)
        reasons = collections.Counter((t['kind'], (t['reason'] or '')[:40]) for t in tr
                                      if t['kind'] in ('replan', 'cold') and t['reason'])
        secs = collections.defaultdict(float)
        evals = collections.defaultdict(int)
        for t in tr:
            secs[t['kind']] += t['seconds'] or 0.0
            evals[t['kind']] += t['n_evals'] or 0
        print(f"\nN={r['N']} k={r['k']} seed={r['start']}: F={r['fun']:.10f} nit={r['nit']} "
              f"{r['seconds']:.0f}s  {r['message'][:60]}")
        print(f"  solves by kind: {dict(kinds)}")
        print(f"  seconds by kind: " + ', '.join(f'{k2} {v:.0f}' for k2, v in secs.items()))
        print(f"  evaluations by kind: {dict(evals)}")
        for (kind, why), n in reasons.most_common(8):
            print(f"    {n:3d} x {kind:6s} {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-iter', type=int, default=80)
    ap.add_argument('--tabulate-only', action='store_true')
    args = ap.parse_args()
    rows = ledger.read_rows(OUT, dedupe=False)
    if not args.tabulate_only:
        for seed in args.seeds:
            rows.append(run_one(args.N, args.k, seed, args.max_iter))
    tabulate([r for r in rows if r['N'] == args.N and r['k'] == args.k])


if __name__ == '__main__':
    main()
