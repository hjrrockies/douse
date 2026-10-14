"""Is the tension well's slope `C_k` a property of the METHOD or of the domain?

    uv run python -m benchmarks.tension_slope_universality [--target 1e-14] [--window 8]

WHAT IS BEING TESTED. Near a mode the MPS tension is a clipped V, `sigma = min(sigma_base,
C_lam |lam - lam_k|)`, and on the two triangles of `jupyter_notebooks/missing_mode.ipynb` the
slope is constant once written in wavenumber:

    C_k = 2 sqrt(lam) C_lam = 0.23 +- 0.01

over 44 basis configurations, two shapes, and `lam` from 87 to 383. If that constant is a
property of the method it is worth a great deal, because the well's width is then

    w_k = 2 q sigma_base / C_k

and "will the scan step over this mode?" becomes answerable BEFORE the scan, from the grid step
`dk` and a measurement of `sigma_base` alone -- no knowledge of where the modes are. That is what
turns the regularization constant from something to guess into something to solve for.

If instead `C_k` drifts with the geometry, the target has to be re-measured per shape and the
criterion is much weaker. Two triangles cannot tell the difference, which is why this sweeps a
family.

WHY RECTANGLES. `lam_{m,n} = pi^2 (m^2/L^2 + n^2/H^2)` in closed form, so the modes are known
exactly and no FEM reference is needed -- the slope is measured against truth rather than against
another discretization. Aspect ratio is then a clean dial for "how elongated", and the triangles
supply the sharp-corner end of the range.
"""
import argparse
import json
import os
import warnings

import numpy as np

from lappy import basis_plan as BP
from lappy import geometry as geo
from lappy import mps
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse import VertexPolygon

PERIM = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
FINE_N = 251
MASK_HALF_REL = 0.02          # off-resonance mask, as a fraction of lam


def rectangle(ratio, perim=PERIM):
    """`(domain, exact spectrum)` for an `L x H` rectangle of the given perimeter."""
    H = perim/(2*(1 + ratio))
    L = ratio*H
    m = np.arange(1, 60)[:, None]
    n = np.arange(1, 60)[None, :]
    sp = np.sort((np.pi**2*(m**2/L**2 + n**2/H**2)).ravel())
    return geo.Polygon(np.array([0, L, L + 1j*H, 1j*H], dtype=complex)), sp


def triangle(p):
    """One of the study shapes, with no exact spectrum -- `None` means 'find the modes yourself'."""
    return VertexPolygon(3, perim=PERIM)(np.asarray(p, float)), None


def solver_for(dom, window=8, target=1e-14, prec=1e-13):
    lam_max = weyl_est(window, dom)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, lam_max, target=target)
        return MPSEigensolver.from_domain(dom, lam_max=lam_max,
                                          basis=BP.realize(plan, dom), prec=prec), plan, lam_max


def sigma_on(solver, lam):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = solver.tensions(np.asarray(lam, float), n_workers=mps.n_workers_default)
    return np.array([r[0] for r in rows], float)


def modes_in(solver, lam_max, exact):
    """The modes to measure at: exact where known, else located by a dense scan.

    A dense scan rather than `solve_interval`, because this file is measuring the shape of the
    tension curve and must not inherit the search's own blind spots -- the whole point of the
    study is that the search misses minima that are plainly there.
    """
    if exact is not None:
        # DEDUPLICATED. A rectangle's spectrum carries genuine degeneracies (lam_{m,n} = lam_{n,m}
        # on the square), and a repeated level makes the "distance to the nearest other mode" zero,
        # which collapsed the sweep window to nothing -- the square and the 2:1 rectangle both
        # died on that before this line existed.
        e = exact[(exact > 0.15*lam_max) & (exact < 0.85*lam_max)]
        keep = np.concatenate([[True], np.diff(e) > 1e-8*np.maximum(e[1:], 1.0)])
        return e[keep][:6]
    g = np.linspace(0.12*lam_max, 0.88*lam_max, 6000)
    s = sigma_on(solver, g)
    loc = [i for i in range(1, len(s) - 1) if s[i] < s[i - 1] and s[i] < s[i + 1]]
    cand = sorted(loc, key=lambda i: s[i])[:12]
    return np.sort(np.array([g[i] for i in cand if s[i] < 1e-3]))[:6]


def slope_at(solver, lam_k, base, gap):
    """`(C_k, floor, lam0)` from a fine sweep whose WINDOW AND RESOLUTION both follow the well.

    A fixed window cannot serve this measurement. `sigma_base` spans two orders of magnitude
    across these domains, and the flank it has to resolve is `0.9 sigma_base / C_lam` wide -- 0.2
    in `lam` on a triangle and tens on a fat rectangle. The first version used one window of
    `0.35 x` the mode spacing for every case: far too coarse for the triangles, where the flank
    got one or two samples and the fit was discarded for five modes out of six, and unnecessarily
    wide for the rectangles.

    So: start narrow, fit, and let the fitted slope size the next pass, requiring at least ~25
    samples across the flank. `gap` (the distance to the nearest other mode) is the hard ceiling,
    since a window that reaches a neighbour measures the wrong thing.
    """
    half = min(2.5, 0.35*gap)
    C_lam, floor, lam0 = np.nan, np.nan, lam_k
    for _ in range(5):
        n = max(FINE_N, 51)
        fine = np.linspace(lam_k - half, lam_k + half, n)
        s = sigma_on(solver, fine)
        i = int(np.argmin(s))
        lam0, floor = float(fine[i]), float(s[i])
        sel = (s > 10*floor) & (s < 0.5*base) & (np.abs(fine - lam0) > 0)
        if sel.sum() >= 5:
            C_lam = float(np.median(s[sel]/np.abs(fine[sel] - lam0)))
            flank = 0.9*base/C_lam                      # half-width of the V at the plateau
            want = min(2.5*flank, 0.35*gap)
            if 0.6 <= want/half <= 1.6:
                break
            half = float(want)
        else:
            half = min(half/4, 0.35*gap)                # too coarse: zoom in and retry
            if half < 1e-6:
                break
    if C_lam != C_lam:
        return np.nan, floor, lam0
    return 2*np.sqrt(lam0)*C_lam, floor, lam0


def study(name, dom, exact, window, target):
    solver, plan, lam_max = solver_for(dom, window, target)
    lams = modes_in(solver, lam_max, exact)
    if len(lams) < 2:
        print(f'{name:>22}  could not locate enough modes to measure')
        return []
    # off-resonance baseline, masking every mode we know about
    g = np.linspace(0.15*lam_max, 0.85*lam_max, 200)
    ref = exact if exact is not None else lams
    g = g[np.min(np.abs(g[:, None] - ref[None, :]), axis=1) > MASK_HALF_REL*g]
    base = float(np.median(sigma_on(solver, g)))

    rows = []
    for j, lk in enumerate(lams):
        others = np.delete(np.asarray(lams, float), j)
        gap = float(np.min(np.abs(others - lk))) if len(others) else 5.0
        C_k, floor, lam0 = slope_at(solver, float(lk), base, gap)
        if C_k == C_k:
            rows.append(dict(name=name, lam=float(lam0), C_k=float(C_k), floor=float(floor),
                             sigma_base=base, n_basis=int(plan.n_total), area=float(dom.area),
                             perim=float(dom.perimeter)))
    if not rows:
        print(f'{name:>22}  no mode gave a usable flank fit')
        return []
    Cs = np.array([r['C_k'] for r in rows])
    print(f'{name:>22}  area={dom.area:.4f}  n_basis={plan.n_total:>4}  '
          f'sigma_base={base:.2e}  C_k = {Cs.mean():.3f} +- {Cs.std():.3f}  '
          f'C_k/(A/P) = {Cs.mean()/(dom.area/dom.perimeter):.3f}   ({len(Cs)} modes, '
          f'lam in [{rows[0]["lam"]:.0f}, {rows[-1]["lam"]:.0f}])')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=1e-14)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'run', 'missing_mode',
                                                  'slope_universality.json'))
    args = ap.parse_args()

    cases = [(f'rectangle {r:g}:1', *rectangle(r)) for r in (1.0, 1.5, 2.0, 3.0, 5.0)]
    cases += [('triangle (good)', *triangle([0.92915, 0.430004])),
              ('triangle (failing)', *triangle([1.028855, 0.413601]))]
    # THE SCALE TEST, which is what separates a fit from a law. `sigma` is a ratio of norms and
    # so dimensionless; `k` is an inverse length; therefore `C_k = dsigma/dk` is a LENGTH, and
    # `C_k ∝ area` cannot be right dimensionally. Every case above holds the perimeter at 4, so
    # `area` and `area/perimeter` are indistinguishable there -- and only the second is
    # admissible. Scaling one rectangle separates them: under `x -> c x`, `area/perimeter`
    # scales like `c` while `area` scales like `c^2`.
    for pm in (2.0, 4.0, 8.0):
        cases.append((f'rect 3:1 perim {pm:g}', *rectangle(3.0, perim=pm)))
    # AND A SHAPE THAT BREAKS THE CORRELATION. Among the rectangles, "fat" and "large area" move
    # together, so neither can be ruled out. An equilateral triangle at perimeter 4 has area
    # 0.770 -- essentially the 3:1 rectangle's 0.750 -- with an entirely different aspect and
    # corner set. If both land near the same `C_k`, area (not elongation) is the variable.
    s = 4.0/3.0
    eq = geo.Polygon(np.array([0, s, 0.5*s + 1j*s*np.sqrt(3)/2], dtype=complex))
    cases.append(('equilateral triangle', eq, None))

    print(f'C_k = 2 sqrt(lam) dsigma/dlam on the well flank, target={args.target:g} '
          f'window={args.window}\n')
    allrows = []
    for name, dom, exact in cases:
        try:
            allrows += study(name, dom, exact, args.window, args.target)
        except Exception as exc:
            print(f'{name:>22}  {type(exc).__name__}: {str(exc)[:60]}')
    if allrows:
        Cs = np.array([r['C_k'] for r in allrows])
        Ar = np.array([r['area'] for r in allrows])
        print(f'\nACROSS EVERYTHING (n={len(Cs)}):')
        print(f'  C_k        = {Cs.mean():.4f} +- {Cs.std():.4f}   '
              f'[{Cs.min():.3f}, {Cs.max():.3f}]  spread {100*Cs.std()/Cs.mean():.0f}%')
        r = Cs/Ar
        print(f'  C_k / area = {r.mean():.4f} +- {r.std():.4f}   '
              f'[{r.min():.3f}, {r.max():.3f}]  spread {100*r.std()/r.mean():.0f}%')
        r2 = Cs/np.sqrt(Ar)
        print(f'  C_k / sqrt(area) = {r2.mean():.4f} +- {r2.std():.4f}  '
              f'spread {100*r2.std()/r2.mean():.0f}%')
        Pr = np.array([r_['perim'] for r_ in allrows])
        r3 = Cs/(Ar/Pr)
        print(f'  C_k / (area/perim) = {r3.mean():.4f} +- {r3.std():.4f}  '
              f'[{r3.min():.3f}, {r3.max():.3f}]  spread {100*r3.std()/r3.mean():.0f}%'
              f'   <-- the dimensionally admissible one')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(allrows, fh, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
