"""Does a stopped run resume if restarted? `maximize_ratio` from a ledger endpoint, with a per-iterate log.

Restart the optimizer at a ledger row's final point `x` with FRESH run state (a new evaluator,
cold first solve, no curvature history, the default initial radius), on a chosen warm path
(`--path windowed|prefix`) and curvature model (`--qn scalar|sr1|none`, passed to
`maximize_ratio(quasi_newton=...)`: True/'scalar', 'sr1', False). Every accepted iterate's
`F`, mode, cluster, radius, `rho` and split are printed (every `--every`-th; 1 = all). If the
endpoint was a genuine stop, a fresh restart cannot climb either; if it climbs, the stop came
from accumulated run state.

The 2026-10-15 STALL ANALYSIS came from these runs (finding 2, both warm paths agreed to 1e-9
per iterate over 30 iterates, climbing at rho ~0.6):

    uv run python -m benchmarks.diagnostics.replay 0 prefix --path prefix --qn sr1 --max-iter 30
    uv run python -m benchmarks.diagnostics.replay 0 prefix --path windowed --qn sr1 --max-iter 30

and the curvature-model comparison (100 iterates, ~2150 s each, every 10th printed):

    uv run python -m benchmarks.diagnostics.replay 0 prefix --qn sr1 --max-iter 100 --every 10
    uv run python -m benchmarks.diagnostics.replay 0 prefix --qn none --max-iter 100 --every 10
    uv run python -m benchmarks.diagnostics.replay 0 prefix --qn scalar --max-iter 100 --every 10

with end values sr1 3.1981306846, none 3.1981270413, scalar ('bb' then) 3.1981257276. Those
were run before the scalar secant existed, by monkeypatching `SR1.update` ('bb' = the Rayleigh
quotient `B = (s.y/s.s) I`, which is what `ScalarSecant` now is, and 'cap', a per-update growth
cap that never fired on this stretch and tracked stock SR1 exactly). The cap is retired; the
staging outputs are quoted in `docs/stall_analysis_2026_10_15.md`.
"""
import time

import numpy as np

from douse.cluster_opt import maximize_ratio

from benchmarks.diagnostics.endpoint import endpoint_from_args, parser, warm_path

QN = {'scalar': 'scalar', 'sr1': 'sr1', 'none': False}


def replay(ev, p0, k, cons=None, max_iter=30, quasi_newton='scalar', rng=None, **kw):
    """`(result, history)`, history one dict per callback: seconds, F, mode, cluster, delta,
    rho (None when not finite) and split (None outside cluster mode)."""
    hist = []
    t0 = time.perf_counter()

    def cb(p, F, model, delta, rho):
        hist.append(dict(t=time.perf_counter() - t0, F=float(F), mode=model['mode'],
                         cluster=tuple(int(i) for i in model['cluster']), delta=float(delta),
                         rho=(float(rho) if np.isfinite(rho) else None),
                         split=(float(model['split']) if 'split' in model else None)))
    res = maximize_ratio(ev, p0, k=k, max_iter=max_iter, constraints=cons, callback=cb,
                         quasi_newton=quasi_newton, rng=rng, **kw)
    return res, hist


def main():
    ap = parser(__doc__)
    ap.add_argument('--path', choices=('windowed', 'prefix'), default='windowed')
    ap.add_argument('--qn', choices=tuple(QN), default='scalar')
    ap.add_argument('--max-iter', type=int, default=30)
    ap.add_argument('--every', type=int, default=1)
    args = ap.parse_args()
    e = endpoint_from_args(args)
    res, hist = replay(warm_path(e.ev, args.path), e.p, e.k, e.cons, args.max_iter, QN[args.qn])
    for i, h in enumerate(hist):
        if (i + 1) % args.every == 0 or i == len(hist) - 1:
            rho = '  -inf' if h['rho'] is None else f"{h['rho']:+.3f}"
            split = '' if h['split'] is None else f" split={h['split']:.1e}"
            print(f"  it={i + 1:3d} {h['t']:6.0f}s F={h['F']:.10f} {h['mode']:7s} cl={h['cluster']} "
                  f"delta={h['delta']:.2e} rho={rho}{split}", flush=True)
    print(f"END {args.qn}/{args.path}: F={res['fun']:.10f} nit={res['nit']} {res['message']}")
    print("   infeasible", res['n_infeasible'], "unresolved", res['n_unresolved'],
          "refused", res['n_solver_refused'], "repairs", res['n_discontinuity_repairs'])
    h = e.ev.health()
    print("   solves", h.n_solves, "warm", h.n_warm, "warm_refused", h.n_warm_refused,
          "replans", h.n_replans)


if __name__ == '__main__':
    main()
