"""Is the SECOND-ORDER path what collapses these runs? Ablate it on the two clearest cases.

At the endpoints of N=5 `lam_4/lam_1` starts 5 and 6 the first-order model is healthy: a step at
1e-03 is feasible and gives rho = +0.85. Yet both runs ground the radius to 5.68e-15. The one
thing the run does that the probe does not is measure rho against `model_gain2` with its SR1
curvature. If `quasi_newton=False` converges where True collapsed, the endgame problem is the
second-order path, not the cluster model.
"""
import os, sys, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
OUT = '/Users/hringer/projects/python/douse/benchmarks/run/ratio_multistart/ablation'


def one(job):
    N, k, start, qn = job
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
    from douse.cluster_opt import maximize_ratio
    warnings.simplefilter('ignore')
    fam = VertexPolygon(N, perim=PERIM)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    ev = SpectrumEvaluator(fam, n_eigs=k + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*k + 4), replan_tol=1e-9)
    p0 = param_vector(start_shape(N, start, 0), PERIM)
    t0 = time.perf_counter()
    res = maximize_ratio(ev, p0, k=k, max_iter=400, constraints=cons, delta0=0.05,
                         quasi_newton=qn)
    out = dict(N=N, k=k, start=start, quasi_newton=qn, fun=float(res.fun), nit=int(res.nit),
               success=bool(res.success), message=str(res.message),
               resolution=float(getattr(res, 'resolution', float('nan'))),
               n_cluster_steps=int(getattr(res, 'n_cluster_steps', 0)),
               seconds=round(time.perf_counter() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'N{N}_k{k}_s{start}_qn{int(qn)}.json'), 'w') as fh:
        json.dump(out, fh)
    return out


if __name__ == '__main__':
    jobs = [(5, 3, 5, True), (5, 3, 5, False), (5, 3, 6, True), (5, 3, 6, False)]
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as ex:
        for r in ex.map(one, jobs):
            print(f"N{r['N']}_k{r['k']} s{r['start']} quasi_newton={str(r['quasi_newton']):5}: "
                  f"F={r['fun']:.9f} nit={r['nit']:>3} {'OK ' if r['success'] else '   '}"
                  f"{r['message'][:38]:40} {r['seconds']:.0f}s", flush=True)
