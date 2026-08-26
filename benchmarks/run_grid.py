"""The bulk sweep: `max lam_k/lam_1` over N-gons, for a grid of `(N, k)`.

    uv run python -m benchmarks.run_grid [--N 4 5 6] [--k 1 2 3 4 5] [--starts 8] [--workers 5]

Each cell is `benchmarks.ratio_multistart` for one `(N, k)`, which runs its starts on a process
pool and appends to its own resumable ledger in `run/ratio_multistart/N<N>_k<k>.jsonl`. So this
driver is itself resumable at cell granularity AND at start granularity: killing it at any moment
loses at most the starts currently in flight, and re-running the same command picks up exactly
where it stopped.

`k` IS 0-BASED HERE, as everywhere in `douse`: `--k 1` is `lam_2/lam_1`. The printed tables are
1-based, because that is how the results get quoted.

ORDERED CHEAP-FIRST. Cells are sorted by an estimate of their cost (`N`, then `k`), so a run that
is cut short has covered the small cases completely rather than a random scatter. Measured
per-iteration costs, 8-iteration samples: N=4 2.5 s, N=5 1.8 s, N=6 2.3 s, N=8 8.9 s, N=12 6.6 s
-- the driver is the number of full-solve fallbacks rather than `N` itself, which is why N=12 is
cheaper than N=8.

READ `resolution` IN THE OUTPUT, not just `F`. A run reports `first-order optimal to resolution
X`, and X is part of the answer -- see `cluster_opt.maximize_ratio`. At N=4 the exact rectangle
oracle is checked automatically as a lower bound.
"""
import argparse
import datetime as _dt
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='*', default=[4, 5, 6])
    ap.add_argument('--k', type=int, nargs='*', default=[1, 2, 3, 4, 5],
                    help='0-based: 1 is lam_2/lam_1')
    ap.add_argument('--starts', type=int, default=8)
    ap.add_argument('--max-iter', type=int, default=400)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--stop-at', type=str, default=None,
                    help='wall-clock HH:MM (next occurrence) after which no NEW cell is started. '
                         'The cell in flight is allowed to finish, so the ledgers are never left '
                         'mid-write; everything is resumable, so the next run picks up exactly '
                         'where this one stopped.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cells = sorted(itertools.product(args.N, args.k), key=lambda c: (c[0], c[1]))
    deadline = None
    if args.stop_at:
        hh, mm = (int(x) for x in args.stop_at.split(':'))
        now = _dt.datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += _dt.timedelta(days=1)
    print(f'{len(cells)} cell(s), {args.starts} start(s) each, {args.workers} worker(s)'
          + (f', stopping before {deadline:%H:%M}' if deadline else '') + '\n')
    t0 = time.perf_counter()
    for i, (N, k) in enumerate(cells, 1):
        if deadline is not None and _dt.datetime.now() >= deadline:
            print(f'\nreached {deadline:%H:%M}; not starting N={N} lam_{k+1}/lam_1. '
                  f'{len(cells)-i+1} cell(s) left -- re-run the same command to resume.',
                  flush=True)
            break
        cmd = [sys.executable, '-u', '-m', 'benchmarks.ratio_multistart',
               '--N', str(N), '--k', str(k), '--starts', str(args.starts),
               '--max-iter', str(args.max_iter), '--workers', str(args.workers),
               # the window and the eigenvalue count both have to grow with k, or the solver is
               # asked for lam_k out of a window that does not contain it
               '--n-eigs', str(k + 2)]
        print(f'=== [{i}/{len(cells)}] N={N} lam_{k+1}/lam_1   ({time.perf_counter()-t0:.0f}s in)',
              flush=True)
        if args.dry_run:
            print('    ' + ' '.join(cmd[3:]))
            continue
        subprocess.run(cmd, cwd=HERE, check=False)
    print(f'\ngrid finished in {(time.perf_counter()-t0)/3600:.2f} h')


if __name__ == '__main__':
    main()
