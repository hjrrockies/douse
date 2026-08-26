"""`max lam_k/lam_1` over N-gons from MANY starts: the first global statement in this repo.

    uv run python -m benchmarks.ratio_multistart --N 5 --k 2 --starts 12 [--max-iter 400]

WHY. Every claim `douse` has made so far is local -- a trust-region stop is a first-order
certificate at one point, and the second-order probe in `douse.certify` is local too. "The
regular pentagon maximizes `lam_2/lam_1`" and "N=5 beats 35/11 at k=3" are both statements a
single run cannot support. This runs the same optimizer from a spread of starts and reports what
the endpoints agree on.

IT ALSO SETTLES A SPECIFIC OPEN QUESTION. Two N=5 `k=3` runs previously ended at 3.191229 and
3.187127 with visibly different shapes (angles 115/125/91/92/117 against 94/101/92/90/163), both
having hit `max_iter`, so both were lower bounds and neither was converged. One optimum or two
was unresolved. Endpoints are fingerprinted by sorted edge lengths and sorted interior angles --
a congruence invariant, since congruent polygons are isospectral -- and grouped, so the answer
is read off the grouping rather than off two runs eyeballed side by side.

STARTS ARE STAR-SHAPED RANDOM POLYGONS, which is a real spread rather than a jitter of the
regular one, and is simple by construction: sorted random angles about a centre with random
radii cannot self-intersect. A jittered regular N-gon is included as start 0 so the near-regular
basin is always represented.

RESUMABLE. One JSON line per completed run in `run/ratio_multistart/N<N>_k<k>.jsonl`; rerunning
the same command skips the starts already there. These runs take minutes each and there are many.

READ `worst_sigma` IN EVERY ROW. A run whose frozen basis drifted is not a result -- that is what
produced the retracted 3.0717 for a problem whose answer is above 3.18. Rows whose `worst_sigma`
exceeds `--sigma-max` are reported separately and excluded from the grouping.

AND READ `worst_weyl_deficit`, which catches the OTHER failure. `worst_sigma` says how well the
basis resolved what it found; it cannot say that a mode was SKIPPED. N=6 `lam_2/lam_1` start 2
once reported 2.6203171651 -- above the PPW bound, therefore impossible -- at a pristine
sigma = 1e-09, because `lam_2` was missing and the run had been optimizing `lam_3/lam_1` for 92
iterations. A value at or below 0 is clean; at or above ~1 a mode is probably missing. See
`douse.spectral_guard`.
"""
import argparse
import json
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from douse import (ClosingBudget, ConstraintSet, MinEdgeLength, SpectrumEvaluator,
                   VertexPolygon, param_vector)
from douse import degeneracy as dg

from douse.cluster_opt import maximize_ratio

try:
    from .rectangle_exact import best_rectangle
except ImportError:                                                   # pragma: no cover
    best_rectangle = None

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_multistart')


def regular(N, perim=PERIM):
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


def start_shape(N, i, seed0=0):
    """Start `i`: 0 is a lightly jittered regular N-gon, the rest are star-shaped and random.

    Star-shaped means simple, which `VertexPolygon.is_valid` would otherwise have to reject at
    a rate that wastes the whole start budget at larger `N`.
    """
    rng = np.random.default_rng(seed0 + 1000*i + N)
    if i == 0:
        v = regular(N)
        return v + 0.02*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()
    ang = np.sort(rng.uniform(0, 2*np.pi, N))
    while np.min(np.diff(np.concatenate([ang, [ang[0] + 2*np.pi]]))) < 0.25:
        ang = np.sort(rng.uniform(0, 2*np.pi, N))
    rad = rng.uniform(0.55, 1.0, N)
    v = rad*np.exp(1j*ang)
    return v*PERIM/np.abs(np.diff(np.concatenate([v, v[:1]]))).sum()


def fingerprint(v):
    """A congruence invariant: sorted edge lengths and sorted interior angles.

    Congruent polygons are isospectral, so two endpoints agreeing here and in `F` are the same
    answer written twice. It is an invariant rather than a complete one -- two incongruent
    polygons can share both multisets -- so it is used to GROUP candidates, and the group's
    spread in `F` is reported alongside so a false merge would show.
    """
    e = np.sort(dg.edge_lengths(v))
    a = np.sort(np.degrees(dg.interior_angles(v)))
    return np.concatenate([e, a])


def run_one(N, k, i, max_iter, seed0, delta0, n_eigs=None):
    fam = VertexPolygon(N, perim=PERIM)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    ev = SpectrumEvaluator(fam, n_eigs=(n_eigs or k + 2), warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*k + 4), replan_tol=1e-9)
    p0 = param_vector(start_shape(N, i, seed0), PERIM)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, p0, k=k, max_iter=max_iter, constraints=cons, delta0=delta0)
    v = fam.vertices(res.x)
    return dict(N=N, k=k, start=i, fun=float(res.fun), nit=int(res.nit),
                seconds=round(time.perf_counter() - t0, 1),
                x=[float(t) for t in res.x],
                lams=[float(t) for t in np.asarray(res.lams)],
                edges=[float(t) for t in np.sort(dg.edge_lengths(v))],
                angles=[float(t) for t in np.sort(np.degrees(dg.interior_angles(v)))],
                worst_sigma=float(res.worst_sigma), n_replans=int(res.n_replans),
                worst_weyl_deficit=float(getattr(res, 'worst_weyl_deficit', float('nan'))),
                n_weyl_flags=int(getattr(res, 'n_weyl_flags', 0)),
                n_qn_steps=int(res.n_qn_steps), n_cluster_steps=int(res.n_cluster_steps),
                n_degeneracy_moves=int(res.n_degeneracy_moves),
                resolution=float(getattr(res, 'resolution', float('nan'))),
                success=bool(res.success), message=str(res.message))


def _cell(job):
    """One start, in its own process. Returns a ledger row, never raises.

    THREADS ARE THROTTLED PER WORKER OR THE POOL IS SLOWER THAN SERIAL. `lappy` evaluates its
    lambda grid on `mps.n_workers_default` threads (8 by default on this machine), and scipy's
    BLAS spawns its own. Five processes each taking 8 threads on ten cores is a 4x
    oversubscription, and oversubscribed threads do not merely fail to help -- they contend for
    the same memory bandwidth the per-sigma LAPACK calls are already limited by (see the note
    above `mps.n_workers_default`). The parent sets the BLAS variables before the pool is created,
    since a spawned child re-imports numpy and reads them then; `n_workers_default` is a module
    global read at call time, so it is set here.
    """
    N, k, i, max_iter, seed0, delta0, n_eigs, threads = job
    from lappy import mps as _mps
    _mps.n_workers_default = max(1, int(threads))
    try:
        return run_one(N, k, i, max_iter, seed0, delta0, n_eigs)
    except Exception as exc:                                          # noqa: BLE001
        return dict(N=N, k=k, start=i, ok=False, error=f'{type(exc).__name__}: {exc}',
                    trace=traceback.format_exc()[-600:])


def group(rows, tol=1e-3):
    """Group rows whose fingerprints agree to `tol` (relative to the perimeter scale)."""
    groups = []
    for r in rows:
        f = np.concatenate([np.asarray(r['edges']), np.asarray(r['angles'])/180.0])
        for g in groups:
            if np.max(np.abs(f - g['f'])) < tol:
                g['rows'].append(r)
                break
        else:
            groups.append(dict(f=f, rows=[r]))
    groups.sort(key=lambda g: -max(r['fun'] for r in g['rows']))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--k', type=int, required=True, help='0-based: k=2 is lam_3/lam_1')
    ap.add_argument('--starts', type=int, default=12)
    ap.add_argument('--max-iter', type=int, default=400)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--delta0', type=float, default=0.05)
    ap.add_argument('--n-eigs', type=int, default=None)
    ap.add_argument('--sigma-max', type=float, default=1e-6)
    ap.add_argument('--workers', type=int, default=1,
                    help='run this many starts concurrently. The cells are independent, so this '
                         'is close to linear until the cores run out; per-worker thread counts '
                         'are throttled to match (see _cell).')
    ap.add_argument('--threads-per-worker', type=int, default=None,
                    help='default: cpu_count // workers, floored at 1.')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'N{args.N}_k{args.k}.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                _r = json.loads(line)
                if _r.get('ok', True):
                    done.add(_r['start'])          # a crashed start is retried on resume
        print(f'resuming {path}: {len(done)} start(s) already recorded', flush=True)

    todo = [i for i in range(args.starts) if i not in done]

    if args.workers > 1 and todo:
        # THE BLAS VARIABLES MUST BE SET BEFORE THE POOL EXISTS. A spawned child re-imports numpy
        # and reads them at import time, so setting them inside the worker is too late.
        threads = args.threads_per_worker or max(1, (os.cpu_count() or 1)//args.workers)
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(var, str(threads))
        jobs = [(args.N, args.k, i, args.max_iter, args.seed0, args.delta0, args.n_eigs, threads)
                for i in todo]
        print(f'running {len(jobs)} start(s) on {args.workers} worker(s), '
              f'{threads} thread(s) each', flush=True)
        # ONE START PER PROCESS. Without `max_tasks_per_child` a worker is recycled across many
        # starts, so each start inherits whatever the previous one's heap left behind -- and a
        # 400-iterate start builds ~400 solvers. `lappy` now bounds and promptly frees its caches,
        # so this is belt and braces rather than the fix, but a fresh process also returns
        # allocator fragmentation to the OS, which nothing inside the process can do. Re-import
        # costs a few seconds against starts that run 20-40 minutes.
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_cell, j): j[2] for j in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                # WRITTEN FROM THE PARENT ONLY, one row at a time, so concurrent workers cannot
                # interleave partial lines into the ledger.
                with open(path, 'a') as fh:
                    fh.write(json.dumps(row) + '\n')
                if not row.get('ok', True):
                    print(f'  start {row["start"]:3d}: FAILED  {row["error"][:88]}', flush=True)
                else:
                    print(f"  start {row['start']:3d}: F={row['fun']:.10f} nit={row['nit']} "
                          f"{row['seconds']:.0f}s sigma={row['worst_sigma']:.1e} "
                          f"res={row.get('resolution', float('nan')):.1e} "
                          f"{'OK ' if row['success'] else '   '}{row['message'][:44]}", flush=True)
        todo = []

    for i in todo:
        # ONE BAD START MUST NOT COST THE OTHER NINE. Measured: an `N=5 k=3` sweep died at start 4
        # on `LinAlgError: SVD did not converge` from `lappy.mps.regularize_pencil`, and took
        # starts 5-9 with it -- an hour of compute lost to a numerical failure on ONE shape, after
        # three starts had already agreed on the answer.
        #
        # RECORDED LOUDLY, NOT SWALLOWED. The failure gets its own ledger row with the traceback
        # tail, prints as FAILED, and is counted in the summary. A bare `except: continue` here
        # would turn "six starts crashed" into "six starts agreed", which is the same class of
        # mistake as the `_degeneracy_move` exception that hid an IndexError through three runs.
        try:
            row = run_one(args.N, args.k, i, args.max_iter, args.seed0, args.delta0, args.n_eigs)
        except Exception as exc:
            row = dict(N=args.N, k=args.k, start=i, ok=False,
                       error=f'{type(exc).__name__}: {exc}',
                       trace=traceback.format_exc()[-600:])
            print(f'  start {i:3d}: FAILED  {row["error"][:88]}', flush=True)
            with open(path, 'a') as fh:
                fh.write(json.dumps(row) + '\n')
            continue
        with open(path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(f"  start {i:3d}: F={row['fun']:.10f} nit={row['nit']} {row['seconds']:.0f}s "
              f"sigma={row['worst_sigma']:.1e} res={row['resolution']:.1e} "
              f"{'OK ' if row['success'] else '   '}{row['message'][:44]}", flush=True)

    allrows = [json.loads(l) for l in open(path) if l.strip()]
    failed = [r for r in allrows if not r.get('ok', True)]
    rows = [r for r in allrows if r.get('ok', True)]

    # THE RECTANGLE ORACLE, at N=4 only. `max lam_k/lam_1` over rectangles is exactly solvable
    # (`benchmarks/rectangle_exact.py`), and rectangles are a subfamily of quadrilaterals, so it
    # is a rigorous LOWER bound on the quadrilateral answer at every k. A run that stops below it
    # has not converged, whatever certificate it reported -- and this is the only check available
    # at k beyond 3.
    bound = None
    if args.N == 4 and best_rectangle is not None:
        _, bound = best_rectangle(args.k, PERIM)
        print(f'\nrectangle oracle at k={args.k + 1} (1-based): {bound:.10f} '
              f'-- a lower bound for the quadrilateral answer')
        for r in rows:
            short = bound - r['fun']
            if short > 1e-7:
                print(f"  start {r['start']:3d} ENDED BELOW THE ORACLE by {short:.3e} "
                      f"({r['fun']:.10f}) -- not converged")

    bad = [r for r in rows if r['worst_sigma'] > args.sigma_max]
    good = [r for r in rows if r['worst_sigma'] <= args.sigma_max]
    print(f'\nN={args.N} k={args.k}: {len(good)} usable of {len(allrows)} start(s) '
          f'({len(bad)} excluded on worst_sigma > {args.sigma_max:.0e}, '
          f'{len(failed)} crashed)')
    for r in failed:
        print(f"  CRASHED start {r['start']}: {r.get('error', '?')[:88]}")
    for gi, g in enumerate(group(good)):
        funs = np.array([r['fun'] for r in g['rows']])
        conv = sum(r['success'] for r in g['rows'])
        print(f'  group {gi}: F = {funs.max():.10f}  (spread {np.ptp(funs):.2e}, '
              f'{len(g["rows"])} run(s), {conv} converged)')
        print(f'      angles {np.array2string(np.asarray(g["rows"][0]["angles"]), precision=2)}')
        print(f'      edges  {np.array2string(np.asarray(g["rows"][0]["edges"]), precision=4)}')
    for r in bad:
        print(f"  EXCLUDED start {r['start']}: worst_sigma={r['worst_sigma']:.1e} F={r['fun']:.8f}")


if __name__ == '__main__':
    main()
