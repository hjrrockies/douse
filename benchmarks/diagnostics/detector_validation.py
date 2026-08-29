"""Does the discontinuity detector repair the three REAL cases it was written from?

    N5_k3 s5   collapsed to 5.68e-15 at F=4.375460168 (trace F 4.441676010, gap 6.6e-02)
    N5_k3 s6   collapsed to 5.68e-15 at F=4.435376453 (trace F 4.455138373, gap 2.0e-02)
    N6_k1 s2   the 2.6203 row; endpoint F gap exactly zero, so ONLY this signature sees it

The cell answers, for reference: N5_k3 -> 4.4823516927, N6_k1 -> 2.5340067775.
"""
import os, sys, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
OUT = '/Users/hringer/projects/python/douse/benchmarks/run/ratio_multistart/detector'
TRUTH = {(5, 3): 4.4823516927, (6, 1): 2.5340067775}


def one(job):
    N, k, start = job
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_v, '2')
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
    t0 = time.perf_counter()
    res = maximize_ratio(ev, param_vector(start_shape(N, start, 0), PERIM), k=k, max_iter=400,
                         constraints=cons, delta0=0.05)
    out = dict(N=N, k=k, start=start, fun=float(res.fun), nit=int(res.nit),
               success=bool(res.success), message=str(res.message),
               n_discontinuity_repairs=int(getattr(res, 'n_discontinuity_repairs', 0)),
               n_unresolved=int(getattr(res, 'n_unresolved', 0)),
               delta=float(res.delta), seconds=round(time.perf_counter() - t0, 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'N{N}_k{k}_s{start}.json'), 'w') as fh:
        json.dump(out, fh)
    return out


if __name__ == '__main__':
    before = {(5, 3, 5): 4.375460168, (5, 3, 6): 4.435376453, (6, 1, 2): 2.526003767}
    with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as ex:
        for r in ex.map(one, [(5, 3, 5), (5, 3, 6), (6, 1, 2)]):
            b = before[(r['N'], r['k'], r['start'])]
            t = TRUTH[(r['N'], r['k'])]
            print(f"N{r['N']}_k{r['k']} s{r['start']}: {b:.9f} -> {r['fun']:.9f} "
                  f"(cell answer {t:.9f}, off by {abs(r['fun']-t):.2e})  nit={r['nit']} "
                  f"repairs={r['n_discontinuity_repairs']} delta={r['delta']:.1e} "
                  f"{'OK ' if r['success'] else '   '}{r['message'][:34]}", flush=True)
