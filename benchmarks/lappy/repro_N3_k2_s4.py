"""Instrumented re-run of N=3, k=2 (lam_3/lam_1), start 4 -- stopped at the FIRST Weyl refusal.

Question: the ledger row for this start records one Weyl refusal, at a shape whose spectrum came
back looking perfect. WHICH domain did the optimizer visit just before it, and did that one solve
cleanly? The recorded refusal carries the failing `p` but not its predecessor, and the predecessor
is half the story -- "lappy was fine here and silently wrong one step later" is a claim about a
PAIR, not about a shape.

So this re-runs the start under the original conditions and logs every eigenvalue call, stopping
the moment the guard refuses on a DEFICIT, by raising a `BaseException` subclass that
`maximize_ratio`'s `except ShapeRefusal` cannot swallow.

THE ANSWER IS THAT THE RE-RUN DOES NOT GET THERE, and that is the finding. Same seed, same
machine, same thread count, and the trajectory diverges at iterate 3 on a SPURIOUS-value refusal
-- where the optimizer shrinks its trust radius and the accumulated float arithmetic decides by
how much. From there it takes a different path and converges cleanly to the cell's certified
answer 2.8354375913 in 153 iterations, worst Weyl deficit 0.19 over 151 evaluations, never
revisiting the failing region at all.

The consequence for testing is worth stating: a known-bad shape cannot be reached by replaying the
run that found it. It has to be STORED and re-solved directly, which is what the `refusals` records
on a ledger row are for -- and, unlike the trajectory, the shape itself reproduces exactly.

Non-invasive: nothing in `douse` or `lappy` is edited, following `repro_N6_k2_s3.py`. Thread setup
matches the original sweep (workers=5 on 10 cores => 2 threads per worker), which is as close as
float determinism allows and, as above, is not close enough.

    uv run python -m benchmarks.diagnostics.repro_N3_k2_s4
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import json, sys, time, warnings, numpy as np
from lappy import mps as _mps
_mps.n_workers_default = 2

from douse.polygon import VertexPolygon
from douse.evaluate import SpectrumEvaluator, UnresolvedSpectrum
from douse.constraints import ConstraintSet, MinEdgeLength, ClosingBudget
from douse.cluster_opt import maximize_ratio
from douse import degeneracy as dg, spectral_guard

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ratio_multistart import start_shape, param_vector, PERIM              # noqa: E402

N, K, START, MAX_ITER, DELTA0, SEED0 = 3, 2, 4, 400, 0.05, 0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'repro_N3_k2_s4.jsonl')

fam = VertexPolygon(N, perim=PERIM)
cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
ev = SpectrumEvaluator(fam, n_eigs=K + 2, warn_gap=0.0, degenerate_tol=0.0,
                       lam_max_window=max(8, 2*K + 4), replan_tol=1e-9)

log = open(OUT, 'w')
state = dict(it=0, ncall=0, p_acc=None, p_prev=None, lams_prev=None, deficit_prev=None,
             delta=np.nan, rho=np.nan,
             t0=time.perf_counter())


class StopAtRefusal(BaseException):
    """Not an `Exception`, so `maximize_ratio`'s `except ShapeRefusal` cannot swallow it."""


def geom(p):
    v = fam.vertices(np.asarray(p, float))
    a = np.degrees(dg.interior_angles(v))
    e = dg.edge_lengths(v)
    return float(a.max()), float(a.min()), float(e.min()/e.max())


_eig = ev.eigenvalues


def eigenvalues(p, seed=None):
    state['ncall'] += 1
    p = np.asarray(p, float)
    amax, amin, eratio = geom(p)
    pa = state['p_acc']
    step = float('nan') if pa is None else float(np.linalg.norm(p - pa))
    rec = dict(it=state['it'], call=state['ncall'], p=[float(t) for t in p],
               step=step, trial=bool(step > 0), max_angle=amax, min_angle=amin,
               edge_ratio=eratio, n_replans=int(ev.n_replans),
               t=round(time.perf_counter() - state['t0'], 1))
    try:
        lams, solver = _eig(p, seed=seed)
    except UnresolvedSpectrum as exc:
        # TWO KINDS OF `UnresolvedSpectrum`, AND ONLY ONE OF THEM IS THE STORY.
        #   * SPURIOUS -- a returned value is not an eigenvalue at all, caught on tension. This is
        #     the LOUD failure: the optimizer sees a refusal, shrinks, and carries on. Recorded
        #     here and re-raised, because stopping on it ends the walk several iterates short of
        #     the shape the ledger actually flagged.
        #   * DEFICIT -- the values are genuine but not consecutive. That is the SILENT failure
        #     made visible by the guard, and the one this harness exists to pair with its
        #     predecessor.
        # The first version stopped on whichever came first and halted on a spurious refusal at
        # iterate 3, 0.117 away in parameter space from the shape that was wanted.
        spurious = 'spurious_tol' in str(exc)
        rec.update(kind='spurious_refusal' if spurious else 'refusal', error=str(exc)[:400],
                   p_prev=state['p_prev'], lams_prev=state['lams_prev'],
                   deficit_prev=state['deficit_prev'],
                   p_acc=(None if pa is None else [float(t) for t in pa]))
        log.write(json.dumps(rec) + '\n'); log.flush()
        if spurious:
            print(f"  spurious refusal at call {state['ncall']}, iterate {state['it']}, "
                  f"{time.perf_counter() - state['t0']:.0f}s -- continuing", flush=True)
            raise                      # let `maximize_ratio` shrink and keep walking
        print(f"\nREFUSED (deficit) at call {state['ncall']}, iterate {state['it']}, "
              f"{time.perf_counter() - state['t0']:.0f}s\n  {str(exc)[:300]}", flush=True)
        raise StopAtRefusal from exc
    lams = np.asarray(lams, float)
    deficit = float(spectral_guard.weyl_deficit(fam(p), lams))
    rec.update(kind='eval', lams=[float(t) for t in lams], deficit=deficit,
               worst_sigma=float(ev.worst_sigma))
    log.write(json.dumps(rec) + '\n'); log.flush()
    state.update(p_prev=[float(t) for t in p], lams_prev=[float(t) for t in lams],
                 deficit_prev=deficit)
    return lams, solver


ev.eigenvalues = eigenvalues


def cb(p, F, model, delta, rho):
    # NO EXTRA SOLVE HERE. An earlier version re-evaluated the accepted iterate to record its
    # spectrum, which doubled the cost of the run and -- far worse -- advanced the evaluator's
    # continuation seed, so the trajectory being "reproduced" was not the recorded one. The
    # spectrum at every shape is already logged by the wrapper above.
    state.update(it=state['it'] + 1, p_acc=np.asarray(p, float).copy(),
                 delta=float(delta), rho=float(rho))
    amax, amin, eratio = geom(p)
    log.write(json.dumps(dict(kind='iter', it=state['it'], call=state['ncall'], F=float(F),
                              p=[float(t) for t in np.asarray(p, float)],
                              delta=float(delta), rho=float(rho), max_angle=amax,
                              min_angle=amin, edge_ratio=eratio,
                              n_replans=int(ev.n_replans),
                              t=round(time.perf_counter() - state['t0'], 1))) + '\n')
    log.flush()
    if state['it'] % 5 == 0:
        print(f"it {state['it']:>4}  F {F:.9f}  delta {delta:.2e}  rho {rho:+.3f}  "
              f"replans {ev.n_replans}  minang {amin:.1f}  "
              f"{time.perf_counter() - state['t0']:.0f}s", flush=True)


p0 = param_vector(start_shape(N, START, SEED0), PERIM)
state['p_acc'] = p0.copy()
print(f'start shape p0 = {np.round(p0, 6).tolist()}  angles '
      f'{np.round(np.degrees(dg.interior_angles(fam.vertices(p0))), 2).tolist()}', flush=True)
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, p0, k=K, max_iter=MAX_ITER, constraints=cons, delta0=DELTA0,
                             callback=cb)
    print(f"\nRAN TO COMPLETION without a refusal: F={res.fun:.10f} nit={res.nit} "
          f"{res.message}", flush=True)
except StopAtRefusal:
    print('stopped at the first refusal, as intended', flush=True)
log.close()
print(f'wrote {OUT}', flush=True)
