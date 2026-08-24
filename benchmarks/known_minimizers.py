"""Recover the known minimizers of lam_1 at fixed perimeter, from a generic starting shape.

Among N-gons of given perimeter the regular one minimizes lam_1 -- a theorem for n = 3, 4
(Polya-Szego) and open for n >= 5. This is the demonstration; `tests/test_optimize.py` holds the
faster regression version.

    .venv/bin/python -m benchmarks.known_minimizers

Recorded 2026-08-24, starting from shapes well away from the answer:

    triangle       lam_1 - 3 pi^2 = -9.6e-14   sides equal to 2.5e-08, angles 60 +- 1.9e-06
    quadrilateral  lam_1 - 2 pi^2 = -9.3e-13   sides equal to 2.1e-07, angles 90 +- 2.2e-05

for 20 s and 99 s respectively.
"""
import warnings, time
import numpy as np
from douse import (VertexPolygon, param_vector, SpectrumEvaluator, ObjectiveFunction,
                   EigenvalueObjective, descend)

PERIM = 4.0
CASES = {
    'triangle  (-> equilateral)': dict(
        n=3, start=np.array([0.0, 1.5, 0.55 + 1.1j]), exact=3*np.pi**2),
    'quadrilateral (-> square)':  dict(
        n=4, start=np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j]), exact=2*np.pi**2),
}

for name, c in CASES.items():
    fam = VertexPolygon(c['n'], perim=PERIM)
    p0 = param_vector(c['start'], PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=1)
    obj = ObjectiveFunction(ev, EigenvalueObjective(0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.time()
        res = descend(obj, p0, max_iter=200, gtol=1e-6)
        dt = time.time() - t0
    v = fam.vertices(res.x)
    sides = np.abs(np.diff(np.append(v, v[0])))
    n = c['n']
    ang = np.degrees([np.pi - np.angle((v[(i+1) % n]-v[i])/(v[i]-v[i-1])) for i in range(n)])
    print(f'{name}')
    print(f'   {res.message}  ({res.nit} iters, {res.nfev} evals, {dt:.0f} s)')
    print(f'   lam_1 = {res.fun:.12f}   exact = {c["exact"]:.12f}   err = {res.fun-c["exact"]:+.2e}')
    print(f'   sides: max |s - mean| = {np.max(np.abs(sides-sides.mean())):.2e}   '
          f'angles: max |a - {180*(n-2)/n:.0f}| = {np.max(np.abs(ang-180*(n-2)/n)):.2e}')
    print(f'   {ev.n_solves} solves, {ev.n_tracks} tracks, {ev.n_track_refusals} track refusals\n')
