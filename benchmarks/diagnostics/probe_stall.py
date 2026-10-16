"""Is a stopped run at an optimum? The model's predicted gain against the true gain, by radius.

At a ledger endpoint, rebuild the first-order local model (`cluster_opt.local_model`, with the
eps-active cluster tied to the radius as `maximize_ratio` ties it: `eps_abs = delta * scale`),
take its trust-region step (`cluster_opt.trust_region_step`) at radii 1e-2 .. 1e-7, solve at the
stepped point, and report the predicted gain, the actual gain and their ratio `rho`. At a true
local maximum the predicted gain shrinks with the radius (slope `gain/delta` -> 0) and `rho` is
noise; at a point the run merely stopped at, the slope stays O(1e-2) and `rho` stays near 1.

The 2026-10-15 STALL ANALYSIS (`docs/stall_analysis_2026_10_15.md`, finding 2) came from

    uv run python -m benchmarks.diagnostics.probe_stall 0 prefix

(seed 0, prefix path, SR1: model slope 2.2e-2, rho 0.85-1.0 at radii 1e-3..1e-7, so not an
optimum), and the same at `2 windowed2` for finding 1. Its printed output was not kept; the
numbers are quoted in the write-up from the session's plan file.
"""
import time

import numpy as np

from douse.cluster_opt import local_model, trust_region_step

from benchmarks.diagnostics.endpoint import endpoint_from_args, parser, ratio

RADII = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def probe(fam, ev, cons, p, k, radii=RADII, eps_rel=1e-3, rng=None):
    """One record per radius: mode, cluster, predicted `gain`, its slope `gain/delta`, the
    `actual` gain of the solve at `p + d`, and `rho = actual/gain` (None when infeasible or the
    solver refused). The first solve at `p` is cold when `ev` is fresh."""
    rng = np.random.default_rng(0) if rng is None else rng
    lams, solver = ev.eigenvalues(p)
    F0 = ratio(lams, k)
    scale = local_model(fam, p, solver, lams, k=k, eps_rel=eps_rel)['scale']
    lin = cons(p) if cons is not None else None
    out = []
    for delta in radii:
        m = local_model(fam, p, solver, lams, k=k, eps_rel=eps_rel, eps_abs=delta*scale)
        d, gain = trust_region_step(m, delta, rng=rng, lin=lin)
        q = p + d
        rec = dict(delta=float(delta), mode=m['mode'], cluster=tuple(int(i) for i in m['cluster']),
                   gain=float(gain), slope=float(gain/delta), step=float(np.linalg.norm(d)),
                   feasible=bool(fam.is_valid(q) and (cons is None or cons.is_feasible(q))),
                   actual=None, rho=None, lam_moves=None, seconds=None)
        if rec['feasible']:
            t0 = time.perf_counter()
            try:
                lq, _ = ev.eigenvalues(q)
            except Exception as e:                      # a refusal is a result here
                rec['refused'] = f'{type(e).__name__}: {str(e)[:80]}'
            else:
                rec['actual'] = ratio(lq, k) - F0
                rec['rho'] = rec['actual']/gain if gain != 0 else None
                rec['lam_moves'] = [float(t) for t in np.asarray(lq[:len(lams)]) - lams]
            rec['seconds'] = time.perf_counter() - t0
        out.append(rec)
    return F0, out


def main():
    args = parser(__doc__).parse_args()
    e = endpoint_from_args(args)
    F0, recs = probe(e.fam, e.ev, e.cons, e.p, e.k)
    print(f"F={F0:.10f} at the endpoint (ledger {e.row['fun']:.10f})")
    for r in recs:
        rho = '   --' if r['rho'] is None else f"{r['rho']:+.3f}"
        act = '        --' if r['actual'] is None else f"{r['actual']:+.3e}"
        print(f"delta={r['delta']:.0e} {r['mode']:7s} cluster={r['cluster']} slope={r['slope']:.3e} "
              f"gain={r['gain']:+.3e} actual={act} rho={rho} feasible={r['feasible']}")


if __name__ == '__main__':
    main()
