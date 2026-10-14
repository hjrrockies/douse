"""Instrumented reproduction of N=6, k=2 (lam_3/lam_1), start 3 -- the worst_sigma=9.5e-04 row.

Question: `replan_tol=1e-9` fired 3 times and the run still ended at 9.5e-04. WHERE does the
tension the guard sees actually get large -- at accepted iterates, or at rejected trial shapes
that the guard then replans AT, leaving the walk on a plan built off-path?

Non-invasive: nothing in `douse` or `lappy` is edited. Two wrappers:
  * `eigenvalues` -- `ev.worst_sigma` is zeroed before each call and read after, so the value
    read back IS the guard's own `sig` for that call (it is the only writer), pre-replan.
  * `replan`      -- records that the guard fired and the shape it re-planned at.
`callback` marks iterate boundaries and carries `delta`/`rho`, so every call is tagged as the
accepted iterate or a trial.

Thread setup matches the original run (workers=5 on 10 cores => 2 threads per worker) so the
trajectory follows the recorded one as closely as float determinism allows.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import json, sys, time, warnings, numpy as np
from lappy import mps as _mps
_mps.n_workers_default = 2

from douse.polygon import VertexPolygon
from douse.evaluate import SpectrumEvaluator
from douse.constraints import ConstraintSet, MinEdgeLength, ClosingBudget
from douse.cluster_opt import maximize_ratio
from douse import degeneracy as dg
sys.path.insert(0, os.path.join(os.getcwd(), 'benchmarks'))
from ratio_multistart import start_shape, param_vector, PERIM

N, K, START, MAX_ITER, DELTA0, SEED0 = 6, 2, 3, 400, 0.05, 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repro_N6_k2_s3.jsonl')

fam = VertexPolygon(N, perim=PERIM)
cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
ev = SpectrumEvaluator(fam, n_eigs=K + 2, warn_gap=0.0, degenerate_tol=0.0,
                       lam_max_window=max(8, 2*K + 4), replan_tol=1e-9)

log = open(OUT, 'w')
state = dict(it=0, ncall=0, p_acc=None, delta=np.nan, rho=np.nan, worst=0.0, t0=time.perf_counter())


def geom(p):
    v = fam.vertices(np.asarray(p, float))
    a = np.degrees(dg.interior_angles(v))
    e = dg.edge_lengths(v)
    return float(a.max()), float(a.min()), float(e.min()/e.max())


_eig, _replan = ev.eigenvalues, ev.replan


def eigenvalues(p, seed=None):
    ev.worst_sigma = 0.0                       # the guard is the only writer; read it back below
    r0, state['ncall'] = ev.n_replans, state['ncall'] + 1
    lams, solver = _eig(p, seed=seed)
    sig = float(ev.worst_sigma)
    state['worst'] = max(state['worst'], sig)
    amax, amin, eratio = geom(p)
    pa = state['p_acc']
    step = float('nan') if pa is None else float(np.linalg.norm(np.asarray(p, float) - pa))
    log.write(json.dumps(dict(
        kind='eval', it=state['it'], call=state['ncall'], sig=sig,
        replanned=int(ev.n_replans - r0), n_replans=int(ev.n_replans), worst=state['worst'],
        step=step, trial=bool(step > 0), delta=state['delta'], rho=state['rho'],
        lam=[float(t) for t in np.asarray(lams)], max_angle=amax, min_angle=amin,
        edge_ratio=eratio, t=round(time.perf_counter() - state['t0'], 1))) + '\n')
    log.flush()
    return lams, solver


def replan(p=None):
    amax, amin, eratio = (geom(p) if p is not None else (float('nan'),)*3)
    log.write(json.dumps(dict(kind='replan', it=state['it'], call=state['ncall'],
                              at_trial=bool(p is not None and state['p_acc'] is not None
                                            and not np.allclose(np.asarray(p, float),
                                                                state['p_acc'])),
                              max_angle=amax, min_angle=amin, edge_ratio=eratio)) + '\n')
    log.flush()
    return _replan(p)


ev.eigenvalues, ev.replan = eigenvalues, replan


def cb(p, F, model, delta, rho):
    state.update(it=state['it'] + 1, p_acc=np.asarray(p, float).copy(), delta=float(delta),
                 rho=float(rho))
    amax, amin, eratio = geom(p)
    log.write(json.dumps(dict(kind='iter', it=state['it'], call=state['ncall'], F=float(F),
                              delta=float(delta), rho=float(rho), max_angle=amax,
                              min_angle=amin, edge_ratio=eratio, worst=state['worst'],
                              n_replans=int(ev.n_replans),
                              t=round(time.perf_counter() - state['t0'], 1))) + '\n')
    log.flush()
    if state['it'] % 10 == 0:
        print(f"it {state['it']:>4}  F {F:.9f}  delta {delta:.2e}  rho {rho:+.3f}  "
              f"worst_sig {state['worst']:.1e}  replans {ev.n_replans}  "
              f"maxang {amax:.1f}  {time.perf_counter() - state['t0']:.0f}s", flush=True)


p0 = param_vector(start_shape(N, START, SEED0), PERIM)
state['p_acc'] = p0.copy()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    res = maximize_ratio(ev, p0, k=K, max_iter=MAX_ITER, constraints=cons, delta0=DELTA0,
                         callback=cb)
print(f"\nDONE  F={res.fun:.10f}  nit={res.nit}  worst_sigma={res.worst_sigma:.3e}  "
      f"n_replans={res.n_replans}  {res.message}", flush=True)
log.close()
