"""Is the returned set consecutive? Weyl against a CHEAP FEM, on the same sets.

    uv run python -m benchmarks.completeness_detectors [--ratios 1 1.633 3 5] [--ks 8 20]
                                                       [--hfrac 60]

WHY A SECOND DETECTOR AT ALL. `spectral_guard.weyl_deficit` compares an ASYMPTOTIC formula against
the returned count, with no error control, and `benchmarks/deficit_calibration.py` measures what
that costs: it refuses CLEAN, exactly-known spectra from k=8 up (earlier when elongated), and past
k ~ 6 no threshold separates clean from missed-mode at all. The statistic is not repairable by
recalibration, so the question is what to put beside it.

WHY FEM IS THE RIGHT SECOND DETECTOR, and it is not "another estimate". Conforming P1 elements on
an EXACTLY triangulated polygon -- a polygon has no geometry error, unlike a curved domain -- span
a subspace of `H^1_0`, so Rayleigh-Ritz gives

    lam_j <= lam_j^h    for every j,    hence    N^h(c) <= N(c)  for every cut c.

The FEM count is therefore a RIGOROUS LOWER BOUND on the true count, and

    N^FEM(cut) > #(values found at or below cut)   PROVES a mode is missing.

A false positive is impossible up to assembly and ARPACK tolerance; a coarse mesh costs
sensitivity, not correctness. That is the exact opposite of the Weyl statistic's failure mode, and
it needs NO calibrated constant -- the deficit is an integer and the threshold is `> 0`.

MEASURED (2026-08-29), exact rectangle spectra, mesh at diameter/60, k=8 and k=20:

    L/H     dofs   Weyl on a CLEAN set   FEM on a CLEAN set   FEM detects   Weyl detects
    1.000   2083   passes                passes               15/15         15/15
    1.633   1865   REFUSES (false +)     passes               28/28         28/28
    3.000   1187   REFUSES (false +)     passes               24/24         24/24
    5.000    761   REFUSES (false +)     passes               27/27         27/27

Same detection, no false positives, 0.0-0.1 s and under 2100 dofs per check.

MESH SIZING IS A RESOLUTION QUESTION, NOT A DIAMETER ONE. At `diameter/60` the thin rectangles
lose sensitivity -- 6 of 8 at 5:1, k=8, because the diameter is the LONG side while the mode
oscillates across the short one, leaving 761 dofs. At `diameter/150` every cell returns to
detecting all of them, at 4873-7708 dofs and 0.2-0.4 s. False positives stay at zero throughout,
which is the point: a coarse mesh costs sensitivity and can never cost correctness. Size the mesh
from the wavelength at `lam_max` rather than from a fixed fraction of the diameter. On the square at K=12
the FEM count also catches all four MISSING-COPY-OF-A-DEGENERATE-LEVEL deletions (`fem_deficit`
+1 each) -- the failure a per-index value test provably cannot see, because the returned VALUES
are unchanged, and the one the N=5 `lam_6/lam_1` cell is suspected of (`estimate_multiplicity`
losing one copy of a double eigenvalue). Counting with multiplicity is what buys that.

WHAT THIS IS NOT. It is not a completeness CERTIFICATE: passing proves nothing, since the bound is
one-sided. A two-sided version exists in the literature -- nonconforming Crouzeix-Raviart elements
give guaranteed LOWER bounds on eigenvalues with explicit constants -- and would turn this into a
proof that the set IS consecutive. Not built, and the constant needs care.

PREREQUISITE: `LAPPY_WISHLIST.md` section 9b. `polygon_triangular_mesh` clamps the mesh floor at
0.05 and `FEMEigensolver` never overrides it, so every request below that silently returns the
SAME mesh. This file works around it explicitly; production use should fix it at the source.
"""
import argparse
import time
import warnings

import numpy as np

from lappy import geometry as geo
from lappy import quad as _quad
from lappy.fem import FEMEigensolver

from douse import spectral_guard

PERIM = 4.0


def _unclamped_mesh():
    """Wishlist 9b: drive the gmsh floor from the requested size instead of the 0.05 default."""
    real = _quad.polygon_triangular_mesh
    if getattr(real, '_unclamped', False):
        return
    def patched(vertices, mesh_size, mesh_size_min=None, mesh_size_max=0.5):
        return real(vertices, mesh_size, mesh_size_min=mesh_size/3, mesh_size_max=mesh_size_max)
    patched._unclamped = True
    _quad.polygon_triangular_mesh = patched


def rectangle(ratio, perim=PERIM):
    """`(domain, exact spectrum)` -- closed form, so "clean" means exactly consecutive."""
    H = perim/(2*(1 + ratio))
    L = ratio*H
    m = np.arange(1, 90)[:, None]
    n = np.arange(1, 90)[None, :]
    sp = np.sort((np.pi**2*(m**2/L**2 + n**2/H**2)).ravel())
    return geo.Polygon(np.array([0, L, L + 1j*H, 1j*H], dtype=complex)), sp


def fem_spectrum(dom, hfrac, lam_top):
    """FEM eigenvalues WITH multiplicity, and the dof count. Every value is an upper bound."""
    _unclamped_mesh()
    fem = FEMEigensolver.from_domain(dom, mesh_size=dom.diameter/hfrac)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eigs, mults, _ = fem.solve_interval(1.0, 4.0*lam_top)
    return (np.sort(np.repeat(np.asarray(eigs, float), np.asarray(mults, int))),
            int(fem._N_dof))


def fem_deficit(fem_lams, lams):
    """Worst `N^FEM(cut) - #found(cut)` over mid-gap cuts. Any value > 0 PROVES a missing mode.

    Same cut placement as `spectral_guard.weyl_deficit`, so this is a drop-in replacement for its
    count -- a rigorous lower bound where that uses an asymptotic formula. Integer valued: there
    is no threshold to calibrate.
    """
    lams = np.asarray(lams, dtype=float)
    lev = spectral_guard.distinct_levels(lams)
    worst = 0
    for a, b in zip(lev[:-1], lev[1:]):
        c = 0.5*(a + b)
        worst = max(worst, int(np.sum(fem_lams <= c)) - int(np.sum(lams <= c)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='*', default=[1.0, 1.633, 3.0, 5.0])
    ap.add_argument('--ks', type=int, nargs='*', default=[8, 20])
    ap.add_argument('--hfrac', type=float, default=60,
                    help='mesh size as diameter/hfrac')
    args = ap.parse_args()

    print(f'{"L/H":>6} {"k":>4} {"dofs":>6} {"s":>5} | {"Weyl on clean":>17} | '
          f'{"FEM on clean":>13} | {"FEM":>9} | {"Weyl":>9}')
    for ratio in args.ratios:
        dom, sp = rectangle(ratio)
        for k in args.ks:
            lams = sp[:k + 2]
            t0 = time.perf_counter()
            fl, ndof = fem_spectrum(dom, args.hfrac, sp[k + 6])
            dt = time.perf_counter() - t0
            if len(fl) < k + 3:
                print(f'{ratio:>6.3f} {k:>4} {ndof:>6}  too few FEM modes -- refine')
                continue

            w_clean = spectral_guard.weyl_deficit(dom, lams) > spectral_guard.DEFICIT_TOL
            f_clean = fem_deficit(fl, lams) > 0

            top = spectral_guard.distinct_levels(lams)[-1]
            nf = nw = tot = 0
            for j in range(1, len(lams)):
                if lams[j] >= top*(1 - 1e-9):
                    continue                 # top gap: above every cut, the documented blind spot
                cut = np.sort(np.concatenate([lams[:j], lams[j+1:], sp[len(lams):len(lams)+1]]))
                tot += 1
                nf += fem_deficit(fl, cut) > 0
                nw += spectral_guard.weyl_deficit(dom, cut) > spectral_guard.DEFICIT_TOL
            print(f'{ratio:>6.3f} {k:>4} {ndof:>6} {dt:>5.1f} | '
                  f'{("REFUSES (false +)" if w_clean else "passes"):>17} | '
                  f'{("REFUSES (false +)" if f_clean else "passes"):>13} | '
                  f'{nf:>4}/{tot:<4} | {nw:>4}/{tot:<4}')


if __name__ == '__main__':
    main()
