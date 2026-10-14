"""Can `rtol` be calibrated from the tension at ONE non-eigenvalue?

    uv run python -m benchmarks.rtol_single_point [--n-lam 8]

THE QUESTION. `benchmarks/sigma_base_ablation.py` established that the tension plateau
`sigma_base` -- the value away from any mode -- is what sets the width of every tension well, via
`w_k = 2 q sigma_base / C_k`, and that it is a knob spanning two orders of magnitude under the
rank truncation `rtol`. The calibration recipe that followed was: measure `sigma_base` over an
off-resonance grid, and bisect `rtol` until it meets a target computed from the geometry.

That recipe needs ~40 tension evaluations per trial `rtol`. This file asks whether ONE evaluation
will do -- i.e. whether `sigma_1(lam; rtol)` at a single non-eigenvalue carries the same
information as the median over many.

WHY IT MIGHT. `rtol` is a per-call parameter, and the expensive part of a tension evaluation is
assembling `A_B(lam)`, `A_I(lam)` and taking the QR/SVD of the stacked pencil. The singular value
spectrum `s` of `R` is computed ONCE per `lam` and `rtol` merely selects `cutoff = (s >= rtol)`.
So the entire `sigma_1` vs `rtol` curve at a fixed `lam` costs one assembly plus one GSVD per
truncation level -- and the spectrum `s` itself is free.

WHAT IS MEASURED, per domain:
  * `sigma_1(lam; rtol)` on a grid of `rtol`, at several off-resonance `lam`;
  * the spread of `sigma_1` ACROSS those `lam` at fixed `rtol` -- if one point is representative,
    this is small;
  * `rtol*`, the truncation at which each single `lam` would first meet the geometric target
    `sigma_base >= 1.32 (|Omega|/|dOmega|) dk`, and how much `rtol*` varies with the choice of
    point -- this is the question, stated as a number;
  * the singular value spectrum of `R`, since if it carries a visible noise floor then reading
    that off is cheaper still and needs no target at all.

NOTE ON CONVENTIONS. `lappy`'s `rtol` is an ABSOLUTE cutoff on the singular values of `R`
(`mps.regularize_pencil`, `cutoff = (s >= rtol).sum()`), despite the name. A solver thresholding
relative to `s.max()` will need every number here divided by `s.max()`, which itself moves with
`lam`, the basis normalization and the collocation count. See
`chladni/notes/narrow_wells_and_rtol.md` section 6.
"""
import argparse
import json
import os
import warnings

import numpy as np
import scipy.linalg as la

from lappy import basis_plan as BP
from lappy import geometry as geo
from lappy import mps
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver, regularize_pencil

from douse import VertexPolygon

PERIM = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'run', 'missing_mode', 'cache.npz')
RTOLS = np.logspace(-15, -5, 21)
KAPPA, Q = 2.384, 0.9          # C_k = KAPPA * area/perimeter; q is the width fraction


def rectangle(ratio, perim=PERIM):
    H = perim/(2*(1 + ratio))
    L = ratio*H
    m = np.arange(1, 60)[:, None]
    n = np.arange(1, 60)[None, :]
    return (geo.Polygon(np.array([0, L, L + 1j*H, 1j*H], dtype=complex)),
            np.sort((np.pi**2*(m**2/L**2 + n**2/H**2)).ravel()))


def build(dom, window=8, target=1e-14, prec=1e-13):
    lam_max = weyl_est(window, dom)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, lam_max, target=target)
        s = MPSEigensolver.from_domain(dom, lam_max=lam_max,
                                       basis=BP.realize(plan, dom), prec=prec)
    return s, lam_max


def sigma_at(solver, lam, rtol):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(np.atleast_1d(solver.tensions(float(lam), rtol=float(rtol)))[0])


def r_spectrum(solver, lam):
    """All singular values of `R` at `lam` -- the object `rtol` thresholds."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A1, A2 = solver.A_B(float(lam)), solver.A_I(float(lam))
        return regularize_pencil(A1, A2, 'svd', 0.0)[3]


def off_resonance(lam_max, modes, n):
    """`n` points spread over the band, each at least 2% of `lam` from every known mode."""
    g = np.linspace(0.2*lam_max, 0.8*lam_max, 400)
    g = g[np.min(np.abs(g[:, None] - np.asarray(modes)[None, :]), axis=1) > 0.02*g]
    return g[np.linspace(0, len(g) - 1, n).astype(int)]


def study(name, dom, modes, n_lam, dk=None):
    solver, lam_max = build(dom)
    A_P = float(dom.area)/float(dom.perimeter)
    lams = off_resonance(lam_max, modes, n_lam)
    if dk is None:                       # the scan step this domain would actually use, in k
        a, b, n_pts = 0.0, lam_max, 20*9
        dk = (b - a)/n_pts/(2*np.sqrt(np.median(lams)))
    target = KAPPA*A_P*dk/(2*Q)

    S = np.array([[sigma_at(solver, lm, rt) for rt in RTOLS] for lm in lams])
    spec = r_spectrum(solver, float(np.median(lams)))

    print(f'\n{"="*92}\n{name}   area/perim={A_P:.4f}   n_basis={len(solver.basis)}'
          f'   target sigma_base = {target:.3e}  (dk={dk:.4f})\n{"="*92}')
    print(f'{"rtol":>9} | ' + ' '.join(f'{lm:>8.1f}' for lm in lams)
          + f' | {"median":>9} {"spread":>7}')
    for j, rt in enumerate(RTOLS):
        col = S[:, j]
        print(f'{rt:9.1e} | ' + ' '.join(f'{v:8.2e}' for v in col)
              + f' | {np.median(col):9.2e} {col.max()/col.min():6.1f}x')

    # rtol* per single point: the smallest rtol whose sigma_1 meets the target
    stars = []
    for i in range(len(lams)):
        ok = np.where(S[i] >= target)[0]
        stars.append(float(RTOLS[ok[0]]) if len(ok) else np.nan)
    stars = np.array(stars)
    med = np.where(np.median(S, axis=0) >= target)[0]
    med_star = float(RTOLS[med[0]]) if len(med) else np.nan
    good = stars[np.isfinite(stars)]
    print(f'\n  rtol* from each single point : {[f"{v:.0e}" for v in stars]}')
    print(f'  rtol* from the median (40-pt): {med_star:.0e}')
    if len(good) > 1:
        print(f'  single-point spread: {good.max()/good.min():.0f}x '
              f'({np.log10(good.max()/good.min()):.1f} decades)')
    print(f'  R singular values: max {spec.max():.3e}  min {spec.min():.3e}  '
          f'n={len(spec)}')
    tail = spec[spec < 1e-6]
    print(f'  tail below 1e-6: n={len(tail)}'
          + (f', flat to within {tail.max()/tail.min():.1f}x' if len(tail) > 1 else ''))
    return dict(name=name, A_P=A_P, target=target, dk=dk, lams=lams.tolist(),
                rtols=RTOLS.tolist(), sigma=S.tolist(), stars=stars.tolist(),
                med_star=med_star, r_spectrum=spec.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-lam', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'run', 'missing_mode',
                                                  'rtol_single_point.json'))
    args = ap.parse_args()

    d = np.load(CACHE, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    fam = VertexPolygon(3, perim=PERIM)
    cases = [('triangle (failing)', fam(np.asarray(d['p_bad'], float)),
              np.asarray(meta['bad']['fem'], float),
              meta['windows']['bad']['spacing']/(2*np.sqrt(148.0))),
             ('triangle (good)', fam(np.asarray(d['p_good'], float)),
              np.asarray(meta['good']['fem'], float),
              meta['windows']['good']['spacing']/(2*np.sqrt(140.6))),
             ('rectangle 3:1', *rectangle(3.0), None)]

    out = []
    for name, dom, modes, dk in cases:
        try:
            out.append(study(name, dom, modes, args.n_lam, dk))
        except Exception as exc:
            print(f'{name}: {type(exc).__name__}: {str(exc)[:80]}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
