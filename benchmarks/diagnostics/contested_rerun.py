"""Re-run N=5 lam_6/lam_1 starts 0 and 1 -- the contested cell -- on the merged solver.

Both claimed a value ABOVE the cell's certified 6.0656540842 (6.07214685 and 6.07220945, 6e-05
apart from each other) and both were excluded on a Weyl deficit of 0.534 / 0.558, marginally over
a threshold of 0.5 that was calibrated on N=4..6 at k<=5. Two readings, and this settles them:

    the values COME DOWN onto 6.06565    the rows were inflated by a missing mode, as the
                                         mechanism predicts, and the certified answer stands
    the values HOLD at 6.0722            either DEFICIT_TOL misfires at this k, or there is a
                                         better basin -- and the N=5 lam_6/lam_1 answer moves

Two starts, one process each, two threads each, matching how they were produced.
"""
import os, sys, json, time, warnings
from concurrent.futures import ProcessPoolExecutor

OUT = '/Users/hringer/projects/python/douse/benchmarks/run/ratio_multistart/rerun'


def one(start):
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_v, '2')
    from lappy import mps as _mps
    _mps.n_workers_default = 2
    sys.path.insert(0, '/Users/hringer/projects/python/douse/benchmarks')
    from ratio_multistart import run_one
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row = run_one(5, 5, start, max_iter=400, seed0=0, delta0=0.05)
    row['seconds_rerun'] = round(time.perf_counter() - t0, 1)
    return row


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    before = {0: 6.07214685, 1: 6.07220945}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for row in ex.map(one, [0, 1]):
            with open(os.path.join(OUT, 'N5_k5_contested_run3.jsonl'), 'a') as fh:
                fh.write(json.dumps(row) + '\n')
            s = row['start']
            if row.get('ok', True):
                print(f"start {s}: BEFORE {before[s]:.8f} -> AFTER {row['fun']:.10f}  "
                      f"deficit {row.get('worst_weyl_deficit', float('nan')):+.3f}  "
                      f"sigma {row['worst_sigma']:.1e}  nit={row['nit']}  "
                      f"unresolved={row.get('n_unresolved')}  [{row['message'][:44]}]", flush=True)
            else:
                print(f"start {s}: FAILED {row['error'][:90]}", flush=True)
    print('\ncertified answer for the cell: 6.0656540842')
