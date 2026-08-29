"""Phase 0: re-run the 11 max_iter/collapse rows instrumented, and classify them.

Each row gets a per-iterate log (F, delta, rho, mode, m, criticality) and, at the endpoint, the
DISCRIMINATOR the handoff names: does `rho -> 1` as the radius shrinks? A model that is right
and an objective that is accurate must give rho -> 1 for a small enough step. If it does, the
stop was premature and the problem is rate or budget; if rho stays negative, the model is wrong
at that point and Phase 2 has a target.

Also recorded per row: the new guards (`worst_sigma_any`, deficit, refusals), because a marginal
index corruption can masquerade as a model failure -- which is exactly what N6_k1 start 2 did.
"""
import os, sys, json, time, warnings
from concurrent.futures import ProcessPoolExecutor

ROWS = [(6, 1, 6), (5, 3, 4), (5, 2, 3), (5, 2, 7), (6, 2, 6), (5, 4, 0),
        (5, 3, 5), (5, 3, 6), (3, 4, 6), (6, 1, 2), (6, 3, 1)]
OUT = '/Users/hringer/projects/python/douse/benchmarks/run/ratio_multistart/phase0'


def one(job):
    N, k, start = job
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_v, '2')
    import numpy as np
    from lappy import mps as _mps
    _mps.n_workers_default = 2
    sys.path.insert(0, '/Users/hringer/projects/python/douse/benchmarks')
    from ratio_multistart import start_shape, param_vector, PERIM
    from douse import (VertexPolygon, SpectrumEvaluator, ConstraintSet, MinEdgeLength,
                       ClosingBudget)
    from douse.cluster_opt import maximize_ratio, local_model, trust_region_step

    warnings.simplefilter('ignore')
    fam = VertexPolygon(N, perim=PERIM)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    ev = SpectrumEvaluator(fam, n_eigs=k + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*k + 4), replan_tol=1e-9)
    p0 = param_vector(start_shape(N, start, 0), PERIM)
    trace = []

    def cb(p, F, model, delta, rho):
        trace.append(dict(it=len(trace) + 1, F=float(F), delta=float(delta), rho=float(rho),
                          mode=model.get('mode'), m=int(model.get('m', 1)),
                          j=int(model.get('j', 0))))

    t0 = time.perf_counter()
    res = maximize_ratio(ev, p0, k=k, max_iter=400, constraints=cons, delta0=0.05, callback=cb)

    # THE DISCRIMINATOR, at the point the run stopped at.
    probe = []
    try:
        lams, solver = ev.eigenvalues(np.asarray(res.x, float))
        F0 = float(lams[k]/lams[0])
        for d in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            model = local_model(fam, np.asarray(res.x, float), solver, lams, k=k, eps_rel=1e-3)
            step, gain = trust_region_step(model, d)
            ev.reset_continuation()
            lq, _ = ev.eigenvalues(np.asarray(res.x, float) + step)
            Fq = float(lq[k]/lq[0])
            probe.append(dict(delta=d, gain=float(gain), crit=float(gain/d),
                              rho=float((Fq - F0)/gain) if gain > 0 else None))
    except Exception as exc:                                          # noqa: BLE001
        probe.append(dict(error=f'{type(exc).__name__}: {str(exc)[:80]}'))

    out = dict(N=N, k=k, start=start, fun=float(res.fun), nit=int(res.nit),
               message=str(res.message), success=bool(res.success),
               worst_sigma=float(res.worst_sigma),
               worst_sigma_any=float(getattr(res, 'worst_sigma_any', float('nan'))),
               worst_weyl_deficit=float(getattr(res, 'worst_weyl_deficit', float('nan'))),
               n_unresolved=int(getattr(res, 'n_unresolved', 0)),
               n_refusals=len(getattr(res, 'refusals', [])),
               resolution=float(getattr(res, 'resolution', float('nan'))),
               seconds=round(time.perf_counter() - t0, 1), probe=probe, trace=trace)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'N{N}_k{k}_s{start}.json'), 'w') as fh:
        json.dump(out, fh)
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as ex:
        for r in ex.map(one, ROWS):
            tail = ' '.join(f"{p['delta']:.0e}:{p['rho']:+.2f}" for p in r['probe']
                            if p.get('rho') is not None)
            print(f"N{r['N']}_k{r['k']} s{r['start']}: F={r['fun']:.9f} nit={r['nit']} "
                  f"{'OK ' if r['success'] else '   '}{r['message'][:30]:32} "
                  f"deficit={r['worst_weyl_deficit']:+.2f} refus={r['n_unresolved']} "
                  f"| rho at shrinking radius: {tail}", flush=True)
