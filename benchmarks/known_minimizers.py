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


# ── lam_2, and the degeneracy question ──────────────────────────────────────────
#
# Minimizing lam_2 over quadrilaterals of perimeter 4 does NOT give the square. For a rectangle
# L x H with L + H = 2 and L >= H the second mode is (2,1), so
#
#     lam_2 = pi^2 (4/L^2 + 1/H^2),   minimized at   L = 2 * 4^(1/3) / (1 + 4^(1/3))
#
#     L = 1.227023580871,  H = 0.772976419129,  lam_2 = 42.7396543166
#
# against the square's 5 pi^2 = 49.3480220054. Recorded 2026-08-24, from a generic starting
# quadrilateral: reached to |dlam_2| = 3.8e-11 and |dL| = 4.2e-07 in 88 iterations, 179 s.
#
# The reason this one was run: to find out whether an objective on lam_2 walks into a degeneracy,
# since the square carries the double lam_2 = lam_3 = 5 pi^2. It does not. The relative gap
# stays near 0.70 along the accepted path (0.38 counting rejected trial steps), five orders above
# `degenerate_tol`. So the simple-eigenvalue gradient is valid for the whole descent and no
# cluster machinery is needed here.
#
# Where it WOULD be needed, the failure is not graceful. At the square itself, measured against
# the exact splits:
#
#     parameter      true directional derivatives      simple-eigenvalue path
#     p[0]              -14.222,  +14.222                  +14.032,  +14.032
#     p[2]              -32.847,  +32.847                   +9.212,   +9.212
#     p[3]              -29.609,  +29.609                   -4.820,   -4.820
#
# Identical for both modes and wrong by factors of three to six, because
# `eigenfunction_coef(lam, mult=1)` returns an arbitrary vector from the two-dimensional
# eigenspace. Hence `SpectrumEvaluator` raises `DegenerateSpectrum` rather than continuing.
