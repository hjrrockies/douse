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

BOTH GUARDS RUN ON THE CONSOLE SUMMARY, not only in `summarize_grid`. They did not, and the
summary printed 2.6203171651 as `group 0` -- the leading line of the output, above the true
answer, with `0 converged` as the only hint that anything was wrong. `summarize_grid` is the tool
of record and filtered it correctly, but this is the output a person actually watches a sweep
produce, and it led with a number no planar domain can achieve. The gates here are the same ones,
imported from `summarize_grid` rather than restated, so the two cannot drift apart.
"""
import argparse
import json
import os
import pathlib
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from douse import (ClosingBudget, ConstraintSet, Convexity, MinEdgeLength, SpectrumEvaluator,
                   VertexPolygon, param_vector)
from douse import degeneracy as dg
from douse import spectral_guard
from douse.sampling import random_convex

from douse.cluster_opt import maximize_ratio
from douse.evaluate import ShapeRefusal
from douse.chladni_backend import _quiet
from douse import ledger, runs
from douse import spec as spec_mod

try:
    from .rectangle_exact import best_rectangle
except ImportError:                                                   # pragma: no cover
    best_rectangle = None

# THE GUARDS ARE IMPORTED, NOT RESTATED, so this summary and `summarize_grid` cannot disagree
# about what counts as a result. No `except ImportError: None` fallback either -- a guard that can
# silently vanish is the failure this is fixing, so an import error should be loud.
try:
    from .summarize_grid import PPW, row_deficit
except ImportError:                                                   # pragma: no cover
    from summarize_grid import PPW, row_deficit

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_multistart')


# Start shapes and the regular polygon live in `douse.runs` (one definition, 2026-10-14).
regular, start_shape, convex_start_shape = runs.regular, runs.start_shape, runs.convex_start_shape


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


def start_line(row):
    """The live per-start line. Flags a missed mode, which `sigma` is structurally blind to."""
    if not row.get('ok', True):
        return f'  start {row["start"]:3d}: FAILED  {row["error"][:88]}'
    d = row_deficit(row)
    flag = '  ** MISSED MODE? **' if d > spectral_guard.DEFICIT_TOL else ''
    r = ledger.normalize(row)

    def num(key):                       # a schema-2 row writes nan as None
        v = r.get(key)
        return float('nan') if v is None else float(v)
    return (f"  start {row['start']:3d}: F={row['fun']:.10f} nit={row['nit']} "
            f"{row['seconds']:.0f}s bound={num('worst_bound'):.1e} "
            f"res={num('resolution'):.1e} "
            f"{'OK ' if row['success'] else '   '}{row['message'][:44]}{flag}")


def partition(rows, sigma_max):
    """`(crashed, drifted, skipped, good)` -- the same two gates `summarize_grid` applies.

    Split out of `main` so it can be tested. A gate that lives only inside a `main` is a gate no
    test can reach, and this pair went missing here for exactly that reason while
    `summarize_grid` kept applying them.
    """
    rows = [ledger.normalize(r) for r in rows]
    crashed = [r for r in rows if not r.get('ok', True)]
    ok = [r for r in rows if r.get('ok', True)]
    # A ROW WITHOUT A HEALTH STATISTIC IS NEITHER CLEAN NOR DRIFTED, and `nan > x` and
    # `nan <= x` are both False, so such a row used to vanish from every bucket and every count
    # (every chladni row did, 2026-10-14). It is reported as UNCERTIFIED instead.
    uncertified = [r for r in ok if not np.isfinite(_sigma(r))]
    drifted = [r for r in ok if _sigma(r) > sigma_max]
    clean = [r for r in ok if _sigma(r) <= sigma_max]
    skipped = [r for r in clean if row_deficit(r) > spectral_guard.DEFICIT_TOL]
    good = [r for r in clean if not (row_deficit(r) > spectral_guard.DEFICIT_TOL)]
    return crashed, drifted, skipped, good, uncertified


def _sigma(row):
    v = row.get('worst_bound', row.get('worst_sigma'))
    return float('nan') if v is None else float(v)


def run_one(N, k, i, max_iter, seed0, delta0, n_eigs=None, convex=False, max_redraws=3,
            prec=1e-10, backend='chladni', spec=None):
    """One start. A start that cannot be SOLVED AT ALL is redrawn rather than lost.

    Measured over the 18-cell grid: **all nine crashed runs died at the start shape**, before a
    single iterate -- five of them the same N=3 triangle with an 11.17 degree spike, failing
    identically at every k, and two the same N=6 start. They are a property of four shapes, not
    of the cells.

    A start is an arbitrary draw, so insisting on a particular one buys nothing. The redraw is
    DETERMINISTIC (`i + 1000*attempt`, a fresh stream that cannot collide with another start's)
    and every discard is recorded with its shape and its error, which is what keeps this from
    being a silent filter: the start distribution becomes "the sampler conditioned on being
    solvable", which is a statable measure, rather than "whatever survived".

    The probe is an explicit evaluation of `p0` rather than an inspection of where an exception
    came from. Only a failure BEFORE the first iterate is redrawn; one that appears later is the
    optimizer walking somewhere hard and is reported as it always was.
    """
    fam = runs.family_for(N, PERIM)
    # `Convexity` is a RESTRICTION and it is meant to be: it asks the convex question. The
    # answer belongs in its own ledger -- see `--convex` in `main`.
    cons = runs.constraints_for(fam, convex)
    draw = convex_start_shape if convex else start_shape
    redraws = []
    for attempt in range(max_redraws + 1):
        # A FRESH EVALUATOR PER ATTEMPT. Its statistics are running maxima over its own lifetime,
        # so reusing one across a redraw carries the DISCARDED shape's numbers into the run that
        # replaces it. Measured, and it cost a result: N=6 `lam_4/lam_1` start 5 discarded a
        # reflex start whose tension was 5.02e-04, converged from the redraw to 4.5353346506 --
        # the cell's answer to 5e-10, with a first-order certificate at criticality 6.7e-08 --
        # and was then EXCLUDED on `worst_sigma = 5.0e-04`, a number belonging to a shape it never
        # optimized. Its own endpoint tensions are 7.0e-13 to 3.3e-12.
        ev = runs.make_evaluator(N, k, n_eigs=n_eigs, fam=fam, prec=prec, backend=backend)
        idx = i if attempt == 0 else i + 1000*attempt
        p0 = param_vector(draw(N, idx, seed0), PERIM)
        try:
            with _quiet():
                ev.eigenvalues(p0)
            break
        except ShapeRefusal as exc:
            v = fam.vertices(p0)
            redraws.append(dict(start_index=idx, error=f'{type(exc).__name__}: {str(exc)[:120]}',
                                angles=[float(t) for t in np.sort(np.degrees(
                                    dg.interior_angles(v)))],
                                edges=[float(t) for t in np.sort(dg.edge_lengths(v))]))
            ev.reset_continuation()
    t0 = time.perf_counter()
    try:
        res = maximize_ratio(ev, p0, k=k, max_iter=max_iter, constraints=cons, delta0=delta0)
    except ShapeRefusal as exc:
        # A REFUSAL AT THE START SHAPE IS FATAL BY DESIGN -- a run cannot begin from a shape whose
        # spectrum it cannot index -- but the outer handler builds its crash row from the
        # exception alone, and the SHAPES would be lost with the evaluator. They are the whole
        # value of the refusal: `benchmarks/replay_refusal.py` re-solves them, and one such pair
        # is what identified the `lappy` failure as a scan stepping over a tight cluster rather
        # than anything to do with the reentrant corner it was assumed to be. Measured cost of
        # not doing this: cell N6_k3 start 5, lost.
        return dict(schema=ledger.SCHEMA, N=N, k=k, start=i, convex=bool(convex), ok=False,
                    redraws=redraws, error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc()[-600:],
                    seconds=round(time.perf_counter() - t0, 1),
                    deficit_tol=float(spectral_guard.DEFICIT_TOL),
                    **{k2: v2 for k2, v2 in ev.health().as_row().items() if k2 != 'trace'},
                    **spec_mod.provenance(ev, spec))
    v = fam.vertices(res.x)
    # THE ROW IS SCHEMA 2 (`douse.ledger`): the run's identity and provenance, the result, and
    # the backend's health under chladni's names, with the per-solve trace. Readers normalize
    # old rows onto the same names.
    return ledger.result_row(
        res, health=ev.health(), N=N, k=k, start=i, convex=bool(convex), redraws=redraws,
        start_index=(i if not redraws else i + 1000*len(redraws)),
        seconds=round(time.perf_counter() - t0, 1),
        edges=[float(t) for t in np.sort(dg.edge_lengths(v))],
        angles=[float(t) for t in np.sort(np.degrees(dg.interior_angles(v)))],
        deficit_tol=float(spectral_guard.DEFICIT_TOL), **spec_mod.provenance(ev, spec))


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
    threads = job.pop('threads', None)           # set per worker by the parent's environment
    # chladni's search is serial per solve and its LAPACK calls run on the BLAS pool; the parent
    # sets the BLAS thread variables before the pool is created (a spawned child reads them on
    # importing numpy). `threads` is kept in the job for that purpose and for the row.
    try:
        return run_one(**job)
    except Exception as exc:                                          # noqa: BLE001
        return dict(N=job['N'], k=job['k'], start=job['i'], ok=False,
                    error=f'{type(exc).__name__}: {exc}',
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
    ap.add_argument('--spec', default=None,
                    help='a run specification (TOML, `douse.spec`): the cells, starts, seed, '
                         'budget, precision, backend and family come from it, every row cites '
                         'it, and the flags below that it covers are ignored.')
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--k', type=int, default=None, help='0-based: k=2 is lam_3/lam_1')
    ap.add_argument('--starts', type=int, default=12)
    ap.add_argument('--max-iter', type=int, default=400)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--delta0', type=float, default=0.05)
    ap.add_argument('--n-eigs', type=int, default=None)
    ap.add_argument('--sigma-max', type=float, default=1e-6)
    ap.add_argument('--convex', action='store_true',
                    help='search CONVEX N-gons only: adds the `Convexity` constraint and draws '
                         'starts uniformly from convex N-gons (Valtr) instead of star-shaped '
                         'ones. Writes a SEPARATE ledger, N<N>_k<k>_convex.jsonl -- the two are '
                         'answers to different questions and must never be pooled.')
    ap.add_argument('--workers', type=int, default=1,
                    help='run this many starts concurrently. The cells are independent, so this '
                         'is close to linear until the cores run out; per-worker thread counts '
                         'are throttled to match (see _cell).')
    ap.add_argument('--threads-per-worker', type=int, default=None,
                    help='default: cpu_count // workers, floored at 1.')
    args = ap.parse_args()
    spec = None
    if args.spec:
        spec = spec_mod.load(args.spec)
        args.starts, args.seed0, args.max_iter = spec.starts, spec.seed0, spec.max_iter
        args.delta0, args.n_eigs, args.convex = spec.delta0, spec.n_eigs, spec.convex
        args.sigma_max = spec.sigma_max
        cells = list(spec.cells)
        print(f'spec {spec.name} ({spec.sha}, {spec.source}): {len(cells)} cell(s)', flush=True)
    else:
        if args.N is None or args.k is None:
            ap.error('give --N and --k, or --spec')
        cells = [(args.N, args.k)]
    for N, k in cells:
        args.N, args.k = N, k
        _run_cell(args, spec)


def _run_cell(args, spec=None):
    """One `(N, k)` cell: resume its ledger, run the missing starts, print the summary."""
    prec = spec.prec if spec is not None else 1e-10
    backend = spec.backend if spec is not None else 'chladni'
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'N{args.N}_k{args.k}{"_convex" if args.convex else ""}.jsonl')
    done = {r['start'] for r in ledger.read_rows(path, dedupe=False)
            if r.get('ok', True) and r.get('start') is not None}  # a crash is retried on resume
    if done:
        print(f'resuming {path}: {len(done)} start(s) already recorded', flush=True)

    todo = [i for i in range(args.starts) if i not in done]

    if args.workers > 1 and todo:
        # THE BLAS VARIABLES MUST BE SET BEFORE THE POOL EXISTS. A spawned child re-imports numpy
        # and reads them at import time, so setting them inside the worker is too late.
        threads = args.threads_per_worker or max(1, (os.cpu_count() or 1)//args.workers)
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(var, str(threads))
        jobs = [dict(N=args.N, k=args.k, i=i, max_iter=args.max_iter, seed0=args.seed0,
                     delta0=args.delta0, n_eigs=args.n_eigs, convex=args.convex, prec=prec,
                     backend=backend, spec=spec, threads=threads) for i in todo]
        print(f'running {len(jobs)} start(s) on {args.workers} worker(s), '
              f'{threads} thread(s) each', flush=True)
        # ONE START PER PROCESS. Without `max_tasks_per_child` a worker is recycled across many
        # starts, so each start inherits whatever the previous one's heap left behind -- and a
        # 400-iterate start builds ~400 solvers. `lappy` now bounds and promptly frees its caches,
        # so this is belt and braces rather than the fix, but a fresh process also returns
        # allocator fragmentation to the OS, which nothing inside the process can do. Re-import
        # costs a few seconds against starts that run 20-40 minutes.
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_cell, j): j['i'] for j in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                # WRITTEN FROM THE PARENT ONLY, one row at a time, so concurrent workers cannot
                # interleave partial lines into the ledger.
                ledger.write_row(path, row)
                print(start_line(row), flush=True)
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
            row = run_one(args.N, args.k, i, args.max_iter, args.seed0, args.delta0,
                          args.n_eigs, convex=args.convex, prec=prec, backend=backend,
                          spec=spec)
        except Exception as exc:
            row = dict(N=args.N, k=args.k, start=i, convex=bool(args.convex), ok=False,
                       error=f'{type(exc).__name__}: {exc}',
                       trace=traceback.format_exc()[-600:])
            print(f'  start {i:3d}: FAILED  {row["error"][:88]}', flush=True)
            ledger.write_row(path, row)
            continue
        ledger.write_row(path, row)
        print(start_line(row), flush=True)

    allrows = ledger.read_rows(path)             # deduped by start, normalized
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

    # THE SAME TWO GATES `summarize_grid` APPLIES, in the same order. A drifted basis is caught
    # by `worst_sigma`; a SKIPPED mode is not -- the 2.6203 row had sigma = 1.3e-09 and was
    # optimizing `lam_3/lam_1` under the name `lam_2/lam_1`. Both are excluded from the grouping
    # and both are reported by name below, so a start that vanishes from the groups says why.
    failed, bad, skipped, good, uncertified = partition(allrows, args.sigma_max)
    print(f'\nN={args.N} k={args.k}: {len(good)} usable of {len(allrows)} start(s) '
          f'({len(bad)} excluded on worst_sigma > {args.sigma_max:.0e}, '
          f'{len(skipped)} on a missed mode, {len(failed)} crashed, '
          f'{len(uncertified)} without a health statistic)')
    for r in uncertified:
        print(f"  UNCERTIFIED start {r['start']}: no worst_sigma recorded")
    for r in failed:
        print(f"  CRASHED start {r['start']}: {r.get('error', '?')[:88]}")
    groups = group(good)
    for gi, g in enumerate(groups):
        funs = np.array([r['fun'] for r in g['rows']])
        conv = sum(r['success'] for r in g['rows'])
        print(f'  group {gi}: F = {funs.max():.10f}  (spread {np.ptp(funs):.2e}, '
              f'{len(g["rows"])} run(s), {conv} converged)')
        print(f'      angles {np.array2string(np.asarray(g["rows"][0]["angles"]), precision=2)}')
        print(f'      edges  {np.array2string(np.asarray(g["rows"][0]["edges"]), precision=4)}')
    for r in bad:
        print(f"  EXCLUDED start {r['start']}: worst_sigma={r['worst_sigma']:.1e} F={r['fun']:.8f}")
    for r in skipped:
        print(f"  MISSED MODE start {r['start']}: weyl_deficit={row_deficit(r):+.3f} "
              f"F={r['fun']:.8f} -- the tracked set is not consecutive, so this is not "
              f"lam_{args.k + 1}/lam_1")

    # PPW IS A THEOREM AND IT IS THE BACKSTOP, not the primary gate: `lam_2/lam_1 <= 2.5387340`
    # for every planar domain, so a k=1 group above it is wrong whatever it certified and
    # whatever the two gates above concluded. It only speaks at k=1, which is why it cannot
    # replace the Weyl deficit.
    if args.k == 1:
        for gi, g in enumerate(groups):
            top = max(r['fun'] for r in g['rows'])
            if top > PPW + 1e-9:
                print(f'\n  *** group {gi} reports {top:.10f}, ABOVE the Payne-Polya-Weinberger '
                      f'bound {PPW:.7f}. No planar domain can achieve this; the spectrum is '
                      f'wrong (usually a MISSED eigenvalue). ***')


if __name__ == '__main__':
    main()
