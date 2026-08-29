"""Where does the missed-mode guard stop working -- in `k`, and in ELONGATION?

    uv run python -m benchmarks.deficit_calibration [--ks 3 5 8 12 20] [--ratios 1 1.5 2 5]

`douse.spectral_guard.DEFICIT_TOL = 0.5` is calibrated on N=4..6 at k<=5, and its own docstring
says so: "a threshold calibrated on N=4..6 at k<=5 is evidence about N=4..6 at k<=5". This is the
measurement that extends it, and the answer is that the threshold does not survive either axis.

NO EIGENSOLVER IS INVOLVED. Rectangle spectra are `pi^2(m^2/L^2 + n^2/H^2)` in closed form, so
every "clean" set here is EXACTLY the consecutive spectrum and every deletion is exactly one
missing mode. Whatever this finds is a property of the statistic, not of the forward solver --
which is the only way to separate the two questions.

WHAT IT FOUND (2026-08-29).

**The guard refuses CLEAN spectra, and the more elongated the shape the earlier it starts.**
Worst deficit over a complete, exactly-known set:

    L/H      k=3      k=5      k=8     k=12     k=16     k=20
    1.000  +0.319   +0.319   +0.319   +0.319   +0.319   +0.424
    1.633  +0.218   +0.218   +0.577!  +0.577!  +0.577!  +0.577!
    2.000  +0.289   +0.289   +0.393   +0.715!  +0.715!  +1.037!
    5.000  +0.423   +0.423   +1.324!  +1.324!  +1.324!  +1.324!

`!` marks a clean spectrum that `DEFICIT_TOL = 0.5` refuses. Verified cut by cut: at every cut the
number of returned values equals the true count taken from the full exact spectrum, so it is
WEYL'S COUNT that is wrong by up to +1.3, not the set. This is the non-uniformity the guard's
docstring warns about, arriving from the ELONGATION direction rather than from `k`.

**Past k ~ 6 no threshold exists at all.** Clean cases and single-mode deletions overlap, with
top-gap deletions excluded (the documented blind spot, which callers cover by asking for `k+2`):

    k      clean max   deleted min   verdict
    5        +0.572      +0.891      a threshold exists
    6        +0.635      +0.715      a threshold exists, margin 0.08
    8        +1.324      +0.705      OVERLAP -- no threshold can separate them
    20       +1.324      +1.007      OVERLAP

**A STEP STATISTIC WAS TRIED AND DOES NOT RESCUE IT.** A missed mode shifts the count by 1 at
every cut above it, so it is a step, while Weyl's error drifts smoothly across the same cuts --
which suggests testing `max(diff(deficit))` instead of `max(deficit)`. Measured over the same
family: clean steps reach +1.066 while deletions go down to +0.148, so the classes overlap there
too. Recorded so it is not re-proposed on the strength of its plausibility.

**The regular N-gon is not a refuge at high k.** From `ngon_basis_probe`-style solves at the
`SpectrumEvaluator` operating point, the clean deficit at the regular N-gon grows with BOTH N and
k -- 6-gon +0.278, 8-gon +0.393, 10-gon **+0.480** at k=20, against a threshold of 0.5. So at
(N=10, k=20) the guard sits 0.02 from refusing the conjectured ANSWER, on the cleanest shape in
the problem.

WHAT THIS MEANS FOR SCOPE. The forward solver is not what limits `k` -- the square resolves to
15.35 digits against its exact spectrum at k=20, and the basis grows 112 -> 120 columns from k=4
to k=20. The GUARD is what limits it. Raising `DEFICIT_TOL` is not the fix: at k>=8 there is no
value that separates the classes. A usable detector has to calibrate against Weyl's error AT THE
SHAPE (the low end of a returned set is known complete, so the error is measurable there) rather
than against a constant fitted at one corner of the (N, k, elongation) space.
"""
import argparse

import numpy as np

from lappy import geometry as geo
from lappy.asymp import weyl_count_poly

from douse import spectral_guard

PERIM = 4.0


def rectangle(ratio, perim=PERIM):
    """`(domain, exact spectrum)` for the perimeter-`perim` rectangle of aspect `ratio`."""
    H = perim/(2*(1 + ratio))
    L = ratio*H
    m = np.arange(1, 90)[:, None]
    n = np.arange(1, 90)[None, :]
    sp = np.sort((np.pi**2*(m**2/L**2 + n**2/H**2)).ravel())
    dom = geo.Polygon(np.array([0, L, L + 1j*H, 1j*H], dtype=complex))
    return dom, sp


def per_cut_deficits(dom, lams):
    """Weyl's count minus the number found, at the middle of every gap between distinct levels."""
    lev = spectral_guard.distinct_levels(lams)
    return np.array([float(weyl_count_poly(0.5*(a + b), dom)) - int(np.sum(lams <= 0.5*(a + b)))
                     for a, b in zip(lev[:-1], lev[1:])])


def deletions(sp, k):
    """Every single-mode deletion from the first `k+2`, EXCLUDING the topmost level.

    The top gap is above every cut point, so no cut can see a mode missing from it -- a documented
    blind spot, and callers cover it by asking for one more eigenvalue than the objective uses.
    Including those here would flatter the statistic's failure rather than measure it.
    """
    lams = sp[:k + 2]
    top = spectral_guard.distinct_levels(lams)[-1]
    for j in range(1, len(lams)):
        if lams[j] >= top*(1 - 1e-9):
            continue
        yield np.sort(np.concatenate([lams[:j], lams[j+1:], sp[len(lams):len(lams) + 1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', type=int, nargs='*', default=[3, 5, 6, 8, 12, 16, 20, 24])
    ap.add_argument('--ratios', type=float, nargs='*',
                    default=[1.0, 1.2, float(np.sqrt(8/3)), 2.0, 3.0, 5.0])
    args = ap.parse_args()

    fam = [(r,) + rectangle(r) for r in args.ratios]

    print('CLEAN deficits -- every one of these sets is the exact consecutive spectrum')
    print(f'{"L/H":>7} ' + ' '.join(f'{("k=%d" % k):>9}' for k in args.ks))
    for ratio, dom, sp in fam:
        cells = []
        for k in args.ks:
            d = float(per_cut_deficits(dom, sp[:k + 2]).max())
            cells.append(f'{d:>+8.3f}' + ('!' if d > spectral_guard.DEFICIT_TOL else ' '))
        print(f'{ratio:>7.3f} ' + ' '.join(cells))
    print(f'  "!" = a CLEAN spectrum that DEFICIT_TOL={spectral_guard.DEFICIT_TOL} refuses\n')

    print('Does ANY threshold separate clean from missed-mode, at each k?')
    print(f'{"k":>4} {"clean max":>11} {"deleted min":>13} {"margin":>9}   verdict')
    for k in args.ks:
        clean = [float(per_cut_deficits(dom, sp[:k + 2]).max()) for _, dom, sp in fam]
        dele = [float(per_cut_deficits(dom, c).max())
                for _, dom, sp in fam for c in deletions(sp, k)]
        c, d = max(clean), min(dele)
        print(f'{k:>4} {c:>+11.3f} {d:>+13.3f} {d - c:>+9.3f}   '
              + ('a threshold exists' if c < d else 'OVERLAP -- no threshold can separate them'))

    print('\nThe STEP statistic, tried and rejected (a missed mode is a step; Weyl drifts)')
    cs, ds = [], []
    for k in args.ks:
        for _, dom, sp in fam:
            v = per_cut_deficits(dom, sp[:k + 2])
            if len(v) > 1:
                cs.append(float(np.max(np.diff(v))))
            for c in deletions(sp, k):
                v = per_cut_deficits(dom, c)
                if len(v) > 1:
                    ds.append(float(np.max(np.diff(v))))
    print(f'  clean max step {max(cs):+.3f}   deleted min step {min(ds):+.3f}   '
          + ('SEPARATES' if max(cs) < min(ds) else 'OVERLAP -- does not rescue the statistic'))


if __name__ == '__main__':
    main()
