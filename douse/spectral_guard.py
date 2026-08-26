"""Is the tracked set the CONSECUTIVE set it claims to be?

`SpectrumEvaluator` returns `K` eigenvalues and every consumer reads them as `lam_1 .. lam_K`.
Nothing checked that they were consecutive, and one failure mode makes them silently not:

    N=6 `max lam_2/lam_1`, start 2, recorded  lams = [17.082832, 44.762438, 74.090865]
    and a ratio of 2.6203171651 -- ABOVE the Payne-Polya-Weinberger bound 2.5387340 that no
    planar domain can exceed. `lam_2 = 41.49929` was simply ABSENT. The run optimized
    `lam_3/lam_1` for 92 iterations while believing it was optimizing `lam_2/lam_1`.

WHY NOTHING ELSE CATCHES IT.

- `worst_sigma` was a pristine 1e-09. All three recorded values ARE genuine eigenvalues, just
  not consecutive ones. Sigma measures how well the basis resolves what it FOUND; it is
  structurally incapable of noticing that something was skipped.
- `verify_final` re-solves on the SAME frozen plan, so a plan that misses a mode misses it again.
- The frozen-plan drift guard in `evaluate.py` is about accuracy FALLING. This is accuracy
  staying perfect while the INDEX is wrong. Different failure, different detector.
- PPW is a theorem but only constrains `lam_2/lam_1`, so it covers `k=1` and nothing above it.

THE DETECTOR. Weyl's three-term law counts eigenvalues below `lam` on a polygon:

    N(lam) ~ (A*lam - P*sqrt(lam))/(4 pi) + sum_j (pi^2 - a_j^2)/(24 pi a_j)

Place a cut point in the middle of every gap between DISTINCT levels of the returned set, and at
each one compare the number of returned values at or below the cut against `N(cut)`. The worst
shortfall over all cuts is the statistic. A cut lands in a spectral gap, where the true counting
function is flat and Weyl's smooth approximation is at its most trustworthy.

CUTTING IN THE GAPS IS THE WHOLE TRICK, and the first version got it wrong. Comparing counts over
the closed window `[lams[0], lams[-1]]` conflates a missed mode with a TRUNCATED DEGENERATE
CLUSTER at the top: asking for 5 eigenvalues of the square returns `[2,5,5,8,10] pi^2` while the
window genuinely holds six, because `10 pi^2` is a double level with one copy left outside. That
version scored a clean set of 5 at +0.535 and a set with a mode deleted at +0.448 -- the classes
crossed over. This problem drives shapes TOWARD degeneracy, so a truncated top cluster is the
common case, not an edge case.

MEASURED SEPARATION, on the square's exactly known spectrum `pi^2 (m^2+n^2)` and on the nine
certified cells of the 2026-08-25 grid:

    clean sets (exact square, k=3..12; grid cells N=4,5,6 k=1..5)   -- worst +0.319
    the real missed-mode run                                       -- +1.010
    synthetic single-mode deletions from the exact square          -- +0.319 to +1.319

`DEFICIT_TOL = 0.5` sits above every clean case with about 0.18 of margin.

A KNOWN BLIND SPOT, and why it is tolerable. The lowest synthetic deletions land exactly on the
worst clean value, so the classes touch. Those are the cases where the deleted mode lies in the
TOPMOST gap -- above the highest cut point, where no cut can see it. Covering that would require
knowing the multiplicity of the top level, which Weyl cannot supply. It is tolerable because the
guard covers every index strictly below the top level, so a caller that asks for at least one
more eigenvalue than its objective consumes has the modes it actually uses inside the covered
range. `benchmarks/ratio_multistart.py` already requests `n_eigs = k + 2` while the ratio needs
`k + 1`, which is exactly that margin.

THIS IS A HEURISTIC, NOT A THEOREM, and it is reported rather than enforced. Weyl's
`O(sqrt(lam))` error is not uniform, and a threshold calibrated on N=4..6 at k<=5 is evidence
about N=4..6 at k<=5. The statistic is recorded per run the way `worst_sigma` is, so a filter can
act on it and a recalibration can be measured against stored rows rather than guessed at.
"""
import numpy as np

from lappy.asymp import weyl_count_poly

DEFICIT_TOL = 0.5
LEVEL_RTOL = 1e-6            # values within this of each other are one degenerate level


def distinct_levels(lams, rtol=LEVEL_RTOL):
    """The distinct eigenvalue levels in a sorted set, collapsing degenerate copies."""
    lams = np.sort(np.asarray(lams, dtype=float))
    levels = []
    for x in lams:
        if not levels or x > levels[-1]*(1.0 + rtol):
            levels.append(float(x))
    return levels


def weyl_deficit(dom, lams, rtol=LEVEL_RTOL):
    """Worst shortfall between Weyl's count and the returned set, over all mid-gap cut points.

    `dom` is a `lappy` Polygon -- its interior angles carry the corner correction, which is not
    negligible at these `lam`. Returns a float: at or below ~0.3 on a clean consecutive set, near
    or above +1 when a mode is missing below the top level. `nan` when the set has fewer than two
    distinct levels, since without a gap there is nowhere to cut.
    """
    lams = np.sort(np.asarray(lams, dtype=float))
    if lams.size < 2 or not np.all(np.isfinite(lams)) or lams[0] <= 0:
        return float('nan')
    levels = distinct_levels(lams, rtol)
    if len(levels) < 2:
        return float('nan')
    worst = -np.inf
    for lo, hi in zip(levels[:-1], levels[1:]):
        cut = 0.5*(lo + hi)
        found = int(np.count_nonzero(lams <= cut))
        worst = max(worst, float(weyl_count_poly(cut, domain=dom)) - found)
    return float(worst)


def weyl_report(dom, lams, tol=DEFICIT_TOL):
    """`(deficit, message)`. `message` is empty unless the deficit exceeds `tol`."""
    d = weyl_deficit(dom, lams)
    if not np.isfinite(d) or d <= tol:
        return d, ''
    lams = np.sort(np.asarray(lams, dtype=float))
    return d, (f'Weyl count is short by {d:.2f} somewhere in [{lams[0]:.6g}, {lams[-1]:.6g}]: '
               f'more eigenvalues are expected below a gap than were found. The tracked set is '
               f'probably NOT consecutive -- a mode was skipped, so the ratio being optimized is '
               f'not the ratio it is labelled. `worst_sigma` cannot see this; see '
               f'`douse.spectral_guard`.')
