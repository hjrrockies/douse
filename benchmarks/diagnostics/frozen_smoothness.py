"""Are the values on one frozen plan a smooth function of the shape?  (2026-11-08, the lock-in gate)

rho is meaningful below the solves' absolute error only if a frozen plan's discrete switches
(SVD rank, integral-rule sizes, the warm path's windows) do not jump between nearby shapes. Walk
a line through a ledger row's endpoint in 13 equal steps of `h`, warm and cold, and read the
third differences of each eigenvalue: a smooth function has none at this scale, so what remains
is the noise of the solve. Noise far above `prec` names a switch to fix.

    python -m benchmarks.diagnostics.frozen_smoothness LEDGER.jsonl ROW [--prec 1e-9] [--seed 0]
"""
import argparse
import json
import warnings

import numpy as np

from douse import runs
from douse.errors import ShapeRefusal

STEPS = (1e-6, 1e-4)
N_POINTS = 13


def third_differences(values):
    """Relative: `|Delta^3 v|/(8 |v|)`, whose size for iid noise of size s is about s."""
    v = np.asarray(values, float)
    return np.abs(np.diff(v, n=3, axis=0))/(8*np.abs(v[:-3]))


def line(ev, p, d, h, warm, n_keep):
    out, replans0 = [], ev.n_replans
    for j in range(N_POINTS):
        if not warm:
            ev.reset_continuation()
        try:
            out.append(np.asarray(ev.eigenvalues(p + j*h*d)[0][:n_keep], float))
        except ShapeRefusal as exc:
            print(f'  refused at j = {j}: {type(exc).__name__}: {str(exc)[:120]}')
            return None, ev.n_replans - replans0
    return np.array(out), ev.n_replans - replans0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger')
    ap.add_argument('row', type=int)
    ap.add_argument('--prec', type=float, default=1e-9)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    row = [json.loads(line_) for line_ in open(a.ledger)][a.row]
    p, N, k = np.array(row['x'], float), row['N'], row['k']
    rng = np.random.default_rng(a.seed)
    d = rng.normal(size=p.size)
    d /= np.linalg.norm(d)
    ev = runs.make_evaluator(N, k, prec=a.prec)
    ev.degenerate_tol = 0.0
    print(f"N={N} k={k} F={row['fun']:.10f} prec={a.prec:.0e}: {row['message'][:70]}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ev.eigenvalues(p)
        for h in STEPS:
            for warm in (False, True):
                vals, replans = line(ev, p, d, h, warm, k + 2)
                if vals is None:
                    continue
                t = third_differences(vals)
                print(f"  h={h:.0e} {'warm' if warm else 'cold'}  replans {replans}  "
                      f"worst noise/prec per level "
                      f"{np.array2string(t.max(axis=0)/a.prec, precision=2)}")


if __name__ == '__main__':
    main()
