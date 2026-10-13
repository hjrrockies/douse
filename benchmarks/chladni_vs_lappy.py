"""The same ratio optimizations on both backends: `lappy` (SpectrumEvaluator) and `chladni`
(ChladniEvaluator, prec 1e-10).  (2026-10-12/13, overnight)

`maximize_ratio` on N-gons of perimeter 4, maximizing lam_{k+1}/lam_1, from perturbed regular
polygons (jitter 0.08, seeds 0..2), N in {4, 5, 6}, k in {1, 2}. The evaluator settings are the
ones `tests/test_cluster_opt.py` uses (degenerate_tol 0, warn_gap 0 for lappy), with n_eigs =
k + 2. Each run's optimum, gap to the degeneracy, iterations, cluster steps, refusals and wall
time go to one JSONL row, written as it finishes.

    .venv/bin/python benchmarks/chladni_vs_lappy.py [WORKERS]       # default 4; JSONL and a table

Each job runs in a fresh process, and its row records its peak memory (`peak_mb`) and a trace
sampled every 30 s; a job past MEM_CAP is stopped and recorded as an error. Run ONE instance at a
time.
    .venv/bin/python benchmarks/chladni_vs_lappy.py report
"""
import _thread
import json
import os
import resource
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "run", "chladni_vs_lappy.jsonl")
PERIM = 4.0
MEM_CAP = 4 * 2**30   # per job; past it the job is stopped and recorded (2026-10-13: an 8+8-worker
                      # run exhausted 16 GB and the kernel panicked)
CASES = [(N, k, seed) for N in (4, 5, 6) for k in (1, 2) for seed in (0, 1, 2)]


def _perturbed(N, jitter, seed):
    rng = np.random.default_rng(seed)
    v = PERIM / N / (2 * np.sin(np.pi / N)) * np.exp(2j * np.pi * np.arange(N) / N)
    return v + jitter * (rng.normal(size=N) + 1j * rng.normal(size=N)) * np.abs(v).mean()


def _peak():
    """Peak resident memory of this process, bytes (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _watch(trace, stop, every=30.0):
    """Sample the peak every `every` seconds into `trace`; interrupt the job past MEM_CAP."""
    t0 = time.perf_counter()
    while not stop.wait(every):
        trace.append((round(time.perf_counter() - t0), round(_peak() / 2**20)))
        if _peak() > MEM_CAP:
            _thread.interrupt_main()
            return


def run_one(args):
    backend, N, k, seed = args
    warnings.simplefilter("ignore")
    from douse.cluster_opt import maximize_ratio
    from douse.polygon import VertexPolygon, param_vector
    fam = VertexPolygon(N, perim=PERIM)
    if backend == "chladni":
        from douse.chladni_backend import ChladniEvaluator
        ev = ChladniEvaluator(fam, n_eigs=k + 2, prec=1e-10, degenerate_tol=0.0)
    else:
        from douse.evaluate import SpectrumEvaluator
        ev = SpectrumEvaluator(fam, n_eigs=k + 2, warn_gap=0.0, degenerate_tol=0.0)
    t0 = time.perf_counter()
    row = dict(backend=backend, N=N, k=k, seed=seed)
    trace, stop = [], threading.Event()
    threading.Thread(target=_watch, args=(trace, stop), daemon=True).start()
    try:
        res = maximize_ratio(ev, param_vector(_perturbed(N, 0.08, seed), PERIM), k=k,
                             max_iter=80)
        row.update(fun=float(res.fun), gap=float(res.gap), nit=int(res.nit),
                   n_cluster_steps=int(res.n_cluster_steps),
                   n_unresolved=int(res.n_unresolved),
                   n_solver_refused=int(res.n_solver_refused),
                   message=str(res.message)[:200], x=[float(t) for t in res.x])
    except Exception as e:  # noqa: BLE001 -- a crash is a result
        row.update(error=f"{type(e).__name__}: {str(e)[:300]}")
    except KeyboardInterrupt:
        row.update(error=f"MemoryCap: peak {_peak() / 2**30:.1f} GB > {MEM_CAP / 2**30:.0f} GB")
    stop.set()
    row["wall"] = time.perf_counter() - t0
    row["peak_mb"] = round(_peak() / 2**20)
    row["mem_trace"] = trace
    return row


def main(workers=4):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["backend"], r["N"], r["k"], r["seed"]))
    jobs = [(b, N, k, s) for (N, k, s) in CASES for b in ("chladni", "lappy")
            if (b, N, k, s) not in done]
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex, open(OUT, "a") as fh:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            row = f.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['backend']:8s} N={row['N']} k={row['k']} seed={row['seed']} "
                  f"{row.get('fun', 'ERR')} {row['wall']:.0f}s {row['peak_mb']} MB", flush=True)
    report()


def report():
    rows = [json.loads(line) for line in open(OUT)]
    by = {}
    for r in rows:
        by.setdefault((r["N"], r["k"], r["seed"]), {})[r["backend"]] = r
    print(f"\n{'N':>2} {'k':>2} {'seed':>4}  {'lappy fun':>14} {'chladni fun':>14} {'diff':>9}  "
          f"{'lappy s':>7} {'chladni s':>9}  notes")
    for key in sorted(by):
        d = by[key]
        lf, cf = d.get("lappy", {}).get("fun"), d.get("chladni", {}).get("fun")
        diff = (cf - lf) if (lf is not None and cf is not None) else float("nan")
        notes = "; ".join(f"{b}: {d[b]['error'][:60]}" for b in d if "error" in d[b])
        print(f"{key[0]:>2} {key[1]:>2} {key[2]:>4}  {lf if lf is not None else float('nan'):14.10f} "
              f"{cf if cf is not None else float('nan'):14.10f} {diff:9.1e}  "
              f"{d.get('lappy', {}).get('wall', float('nan')):7.0f} "
              f"{d.get('chladni', {}).get('wall', float('nan')):9.0f}  {notes}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
