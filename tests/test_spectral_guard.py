"""The Weyl-count guard: does a NON-consecutive tracked set get caught?

`worst_sigma` is blind to this failure by construction -- it measures how well the basis resolved
what it found, not whether something was skipped -- so these tests are the only thing standing
between a missed mode and a published number. See `douse.spectral_guard`.

Most of these use the unit square, whose spectrum is `pi^2 (m^2 + n^2)` in closed form. No
eigensolver runs, so a failure here is a failure of the guard and never of the solver.
"""
import numpy as np
import pytest

from douse import VertexPolygon, spectral_guard as sg

PERIM = 4.0

# The run that motivated the guard: N=6 `max lam_2/lam_1`, start 2 of the 2026-08-25 grid.
# It reported 2.6203171651, ABOVE the Payne-Polya-Weinberger bound 2.5387340 that no planar
# domain can exceed, at `worst_sigma = 1e-09`. `lam_2 = 41.49929` was missing from the set.
MISSED_X = [0.796102460003, 1.043064923461, 0.304550994148, -0.296494702303,
            0.1143544808, 0.862628738988, 1.224013681815, 0.642456026831]
MISSED_LAMS = [17.082832, 44.762438, 74.090865]

# The certified endpoint of the same cell: the regular hexagon, 2.5340067775.
CLEAN_X = [0.999980341185, 0.666556302539, -0.000119176496, -0.333355366575,
           0.577411043193, 1.154754371783, 1.154648231023, 0.577269874644]
CLEAN_LAMS = [16.099513, 40.796275, 40.796275]


def _hex(x):
    return VertexPolygon(6, perim=PERIM)(np.array(x))


def _square():
    return VertexPolygon(4, perim=PERIM)(np.array([1.0, 0.0, 1.0, 1.0]))


def _square_spectrum(n=8):
    return np.sort([np.pi**2*(m*m + n_*n_) for m in range(1, n) for n_ in range(1, n)])


# ── the two real specimens ───────────────────────────────────────────────────

def test_the_missed_mode_run_is_flagged():
    """THE regression test. This exact shape and this exact set of values went 92 iterations
    optimizing `lam_3/lam_1` while labelled `lam_2/lam_1`, and nothing caught it."""
    d, msg = sg.weyl_report(_hex(MISSED_X), MISSED_LAMS)
    assert d > sg.DEFICIT_TOL, f'the missed mode must be flagged, got deficit {d:.3f}'
    assert msg and 'not consecutive' in msg.lower()


def test_the_certified_endpoint_of_the_same_cell_is_not_flagged():
    """The discriminator has to separate, not just fire. Same cell, same N, same k -- and note
    this set ends on a DEGENERATE pair, which is what broke the first version of the statistic."""
    d, msg = sg.weyl_report(_hex(CLEAN_X), CLEAN_LAMS)
    assert d <= sg.DEFICIT_TOL, f'a clean consecutive set must pass, got deficit {d:.3f}'
    assert msg == ''


def test_the_threshold_sits_between_them_with_margin():
    good = sg.weyl_deficit(_hex(CLEAN_X), CLEAN_LAMS)
    bad = sg.weyl_deficit(_hex(MISSED_X), MISSED_LAMS)
    assert good < sg.DEFICIT_TOL < bad
    assert bad - good > 0.5, (good, bad)


# ── the exactly known spectrum ───────────────────────────────────────────────

@pytest.mark.parametrize('k', [3, 4, 5, 6, 7, 8, 10, 12])
def test_an_exact_consecutive_set_is_never_flagged(k):
    """Includes `k=5` and `k=6`, which end on the double level `10 pi^2`. The first version of
    this statistic scored the clean `k=5` set at +0.535 and flagged it: it was counting a
    truncated cluster as a missing mode."""
    d = sg.weyl_deficit(_square(), _square_spectrum()[:k])
    assert d <= sg.DEFICIT_TOL, f'exact consecutive set of {k} flagged, deficit {d:.3f}'


@pytest.mark.parametrize('k,drop', [(8, 1), (8, 2), (8, 3), (10, 1), (10, 4), (12, 2), (12, 5)])
def test_deleting_a_mode_below_the_top_gap_is_flagged(k, drop):
    """A controlled version of the failure, rather than the one specimen it was found on."""
    punctured = np.delete(_square_spectrum()[:k], drop)
    d, msg = sg.weyl_report(_square(), punctured)
    assert d > sg.DEFICIT_TOL and msg, f'deleting index {drop} of {k} was missed, got {d:.3f}'


def test_a_deletion_in_the_TOPMOST_gap_is_NOT_caught():
    """THE KNOWN BLIND SPOT, pinned deliberately so it is not mistaken for a passing case.

    Cuts are placed in the gaps between returned levels, so nothing can see above the highest
    one. Deleting `8 pi^2` from `[2,5,5,8,10,10] pi^2` leaves the hole above the last cut, and
    the statistic lands at +0.402 -- indistinguishable from a clean set.

    Covering this would need the multiplicity of the top level, which Weyl cannot supply. It is
    tolerable only because a caller that asks for more eigenvalues than its objective consumes
    keeps the indices it USES below the top level; `ratio_multistart` asks for `k+2` and uses
    `k+1`. If that margin is ever removed, this test is the reason it cannot be.
    """
    punctured = np.delete(_square_spectrum()[:6], 3)
    d = sg.weyl_deficit(_square(), punctured)
    assert d <= sg.DEFICIT_TOL, (
        f'blind spot unexpectedly closed (deficit {d:.3f}) -- if this is a real improvement, '
        f'update the module docstring and delete this test')


# ── degenerate input ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('lams', [[], [3.0], [np.nan, 2.0], [-1.0, 2.0], [5.0, 5.0]])
def test_input_without_a_usable_gap_returns_nan_rather_than_a_verdict(lams):
    """With fewer than two distinct levels there is nowhere to cut, and inventing a verdict would
    be worse than declining one. `[5.0, 5.0]` is one degenerate level, not two."""
    assert np.isnan(sg.weyl_deficit(_hex(CLEAN_X), lams))


def test_distinct_levels_collapses_degenerate_copies():
    assert len(sg.distinct_levels([1.0, 2.0, 2.0, 2.0, 3.0])) == 3
    assert len(sg.distinct_levels([1.0, 1.0 + 1e-12])) == 1


# ── the evaluator carries it ─────────────────────────────────────────────────

@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_evaluator_records_the_worst_deficit_over_a_run():
    """`worst_weyl_deficit` has to be tracked the way `worst_sigma` is, or a caller cannot filter
    on it after the fact."""
    from douse import LappySpectrumEvaluator as SpectrumEvaluator
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0, replan_tol=1e-9)
    assert ev.worst_weyl_deficit == -np.inf and ev.n_weyl_flags == 0
    ev.eigenvalues(np.array([1.0, 0.0, 1.0, 1.0]))
    assert np.isfinite(ev.worst_weyl_deficit)
    assert ev.worst_weyl_deficit <= sg.DEFICIT_TOL, 'the square must not flag'
    assert ev.n_weyl_flags == 0
