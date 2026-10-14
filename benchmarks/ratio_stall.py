"""Does the smooth optimizer reach the regular N-gon when maximizing lam_2/lam_1?

    .venv/bin/python -m benchmarks.ratio_stall [N ...]

THE QUESTION. The conjectured maximizer is the regular N-gon, where lam_2 = lam_3 EXACTLY, so
the objective is only directionally differentiable there and a gradient method has no reason to
converge. Before building the cluster/SDP machinery to handle that, it is worth measuring how
badly the smooth method actually fails: if it climbs to within 1e-08 of the regular N-gon and
then chatters, the answer is already in hand and the machinery buys rigour rather than results.
If it stalls at 1e-03, the machinery is required.

RESUMABLE, because this takes the better part of an hour and a laptop may sleep. Every accepted
iterate is appended to `run/ratio_stall/N<k>.jsonl`; restarting the same command picks up from
the last line rather than starting over. macOS suspends rather than kills a sleeping process, so
a lid-close is usually survivable on its own -- this removes the "usually".

`SpectrumEvaluator` is deliberately configured with `degenerate_tol=0` here: the point is to
watch the smooth method walk INTO the degeneracy, so the guard that normally refuses must be
off. The gap is recorded at every step instead.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

from douse import ledger, runs
from douse import (VertexPolygon, param_vector, SpectrumEvaluator, ObjectiveFunction,
                   RatioObjective)

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_stall')
DISK = (3.8317059702/2.4048255577)**2


def regular(N, perim=PERIM):
    return runs.regular(N, perim)


def start_shape(N, jitter=0.13, seed=0):
    """A regular N-gon pushed well off symmetry, so the run has somewhere to travel from."""
    return runs.perturbed(N, jitter, seed)


def _ledger(N):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f'N{N}.jsonl')


def _load(path):
    return ledger.read_rows(path, dedupe=False)


def _append(path, rec):
    ledger.write_row(path, rec)


def run(N, max_iter=400, alpha0=0.01, grow=1.6, shrink=0.5, max_backtrack=40, gtol=1e-9):
    """Backtracking ascent on lam_2/lam_1, checkpointing every accepted step.

    Mirrors `douse.optimize.descend`, rather than calling it, because the checkpoint has to live
    INSIDE the loop -- an optimizer that only reports at the end cannot be resumed.
    """
    path = _ledger(N)
    fam = VertexPolygon(N, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    obj = ObjectiveFunction(ev, RatioObjective(1))          # sign=-1: maximizes the ratio

    done = _load(path)
    if done:
        last = done[-1]
        p, alpha, it0 = np.array(last['p']), last['alpha'], last['it'] + 1
        print(f'N={N}: resuming from iteration {it0} of {path}', flush=True)
    else:
        p, alpha, it0 = param_vector(start_shape(N), PERIM), alpha0, 0
        print(f'N={N}: starting fresh ({fam.n_params} parameters)', flush=True)

    p_reg = param_vector(regular(N), PERIM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, grad = obj(p)

        t0, stop = time.time(), 'reached max_iter'
        for it in range(it0, max_iter):
            gnorm = float(np.linalg.norm(grad))
            if gnorm < gtol:
                stop = f'gradient norm {gnorm:.3g} below gtol'
                break
            stepped = False
            for _ in range(max_backtrack):
                q = p - alpha*grad
                if fam.is_valid(q):
                    v2, g2 = obj(q)
                    if np.isfinite(v2) and v2 < val:
                        p, val, grad, stepped = q, v2, g2, True
                        alpha *= grow
                        break
                alpha *= shrink
            if not stepped:
                stop = f'no decrease after {max_backtrack} backtracks (step {alpha:.3g})'
                break

            lams = ev._last[1]
            rec = dict(it=it, p=list(map(float, p)), alpha=float(alpha),
                       ratio=float(-val), lams=list(map(float, lams)),
                       gap=float(abs(lams[2] - lams[1])/lams[1]),
                       gnorm=float(np.linalg.norm(grad)),
                       dist_to_regular=float(np.linalg.norm(p - p_reg)),
                       seconds=time.time() - t0)
            _append(path, rec)
            if it % 5 == 0 or it < 5:
                print(f'  it {it:4d}  ratio {rec["ratio"]:.10f}  gap {rec["gap"]:.2e}  '
                      f'|g| {rec["gnorm"]:.2e}  |p-reg| {rec["dist_to_regular"]:.2e}',
                      flush=True)

    recs = _load(path)
    if recs:
        r = recs[-1]
        print(f'N={N}: {stop}')
        print(f'  final ratio  {r["ratio"]:.12f}   regular N-gon gives '
              f'{ratio_at_regular(N):.12f}   deficit {ratio_at_regular(N)-r["ratio"]:+.3e}')
        print(f'  final gap lam_2-lam_3 = {r["gap"]:.3e}   |p - p_regular| = '
              f'{r["dist_to_regular"]:.3e}   ({len(recs)} iterations)', flush=True)
    return recs


_RATIO_CACHE = {}


def ratio_at_regular(N):
    """lam_2/lam_1 at the regular N-gon, solved once and remembered."""
    if N not in _RATIO_CACHE:
        from lappy import Eigenproblem
        fam = VertexPolygon(N, perim=PERIM)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            e = np.asarray(Eigenproblem(fam(param_vector(regular(N), PERIM)),
                                        precision=1e-11).solve(2), dtype=float)
        _RATIO_CACHE[N] = float(e[1]/e[0])
    return _RATIO_CACHE[N]


if __name__ == '__main__':
    Ns = [int(a) for a in sys.argv[1:]] or [4, 5]
    print(f'disk (PPW bound): {DISK:.7f}\n')
    for N in Ns:
        run(N)
        print(flush=True)
