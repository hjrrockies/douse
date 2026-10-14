"""The exact rectangle answers, checked against closed forms rather than against themselves.

`benchmarks/rectangle_exact.py` is an ORACLE -- `max lam_k/lam_1` over rectangles is a rigorous
lower bound for the same maximum over quadrilaterals, and it is the only independent check
available at `k` beyond 3. An oracle that is wrong is worse than no oracle, so it is pinned here
against values derived by hand.
"""
import numpy as np
import pytest

from benchmarks.rectangle_exact import best_rectangle, ratio, spectrum


def test_the_square_spectrum_is_the_textbook_one():
    """`pi^2 (m^2 + n^2)`: 2, 5, 5, 8, 10, 10, 13, 13, 17, 17 in units of `pi^2`."""
    s = spectrum(1.0, 1.0, 12)/np.pi**2
    assert s[:10] == pytest.approx([2, 5, 5, 8, 10, 10, 13, 13, 17, 17])


def test_the_k3_optimum_is_the_sqrt_8_3_rectangle_at_35_over_11():
    """The headline N=4 result, independently derived.

    `lam_3 = lam_4` when the (1,2) and (3,1) modes cross: `1/L^2 + 4/H^2 = 9/L^2 + 1/H^2`, i.e.
    `H^2/L^2 = 8/3`. With the perimeter fixed at 4 that gives `lam_3/lam_1 = 35/11` exactly.
    """
    L, val = best_rectangle(2, 4.0)                      # 0-based: index 2 is lam_3
    assert val == pytest.approx(35/11, rel=1e-9)
    H = 2.0 - L
    assert (H/L)**2 == pytest.approx(8/3, rel=1e-6)


def test_the_k2_optimum_is_the_square_at_5_over_2():
    """`max lam_2/lam_1` over rectangles is the square, where `lam_2 = lam_3 = 5 pi^2`."""
    L, val = best_rectangle(1, 4.0)
    assert val == pytest.approx(2.5, rel=1e-10)
    assert L == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('k1,expected', [(6, 77/13), (8, 7.0), (11, 28/3), (13, 11.0),
                                         (17, 69/5), (25, 221/11)])
def test_the_optimal_ratio_is_rational(k1, expected):
    """Every optimum sits at a mode CROSSING, which pins `L^2/H^2` to a rational; the `pi^2/L^2`
    then cancels out of `lam_k/lam_1` and the ratio is rational outright. Measured rational at
    all 29 indices from k=2 to k=30 -- these six are the spot checks."""
    _, val = best_rectangle(k1 - 1, 4.0)
    assert val == pytest.approx(expected, rel=1e-7)


def test_the_optimum_is_never_below_the_square():
    """The square is one competitor among many, so the maximum is at least its value. A grid
    scan that landed in the wrong basin would show up here before it poisoned an oracle."""
    for k1 in range(2, 20):
        L, val = best_rectangle(k1 - 1, 4.0)
        assert val >= ratio(1.0, k1 - 1, 4.0) - 1e-9, (k1, val)


def test_enough_modes_are_taken_that_the_kth_is_present():
    """The truncation `m, n <= k + 12` must not clip the k-th eigenvalue. Compare against a much
    larger truncation rather than trusting the margin."""
    for k in (1, 5, 20, 39):
        for L in (0.5, 0.75, 1.0):
            small = spectrum(L, 2.0 - L, k + 12)[:k + 1]
            large = spectrum(L, 2.0 - L, k + 60)[:k + 1]
            assert small == pytest.approx(large, rel=1e-12)


def test_the_oracle_hits_the_exact_rationals_to_machine_precision():
    """The optimum is a KINK, so a smooth scalar minimizer is not enough on its own.

    `minimize_scalar(method='bounded')` is Brent's method and assumes smoothness; at a crossing it
    stalls, landing systematically LOW -- measured at 2.6e-10 to 5.5e-08 below the true value
    before `_crossing_L` was added. Always low, so the oracle PROPERTY (a lower bound on the
    quadrilateral answer) survived, but "exact" was overstated by eight digits and the comparison
    column carried that as noise. Solving the crossing in closed form fixes it.
    """
    from fractions import Fraction

    exact = {2: Fraction(5, 2), 3: Fraction(35, 11), 4: Fraction(4), 6: Fraction(77, 13),
             8: Fraction(7), 11: Fraction(28, 3), 13: Fraction(11), 17: Fraction(69, 5),
             25: Fraction(221, 11), 28: Fraction(43, 2)}
    for k1, fr in exact.items():
        _, val = best_rectangle(k1 - 1, 4.0)
        assert val == pytest.approx(float(fr), abs=1e-13), (k1, val, float(fr))


def test_the_crossing_solve_declines_when_there_is_no_crossing():
    """Not every optimum is a kink -- the square is a symmetry point -- so the closed-form step is
    accepted only when it does not make the objective worse, and the iterate survives otherwise."""
    from benchmarks.rectangle_exact import _crossing_L, _n_modes_for, ratio

    # k=2 (0-based 1) is the square; the snap must not move it off 5/2
    L, val = best_rectangle(1, 4.0)
    assert val == pytest.approx(2.5, abs=1e-14)
    assert L == pytest.approx(1.0, abs=1e-9)

    # and a crossing solve that returns something must genuinely be a crossing
    for k in (2, 5, 7):
        L_opt, _ = best_rectangle(k, 4.0)
        Lx = _crossing_L(L_opt, k, 4.0, _n_modes_for(k))
        if Lx is None:
            continue
        s = np.sort((np.pi**2*(np.arange(1, _n_modes_for(k) + 1)[:, None]**2/Lx**2
                               + np.arange(1, _n_modes_for(k) + 1)[None, :]**2/(2.0 - Lx)**2)
                     ).ravel())
        assert s[k + 1] - s[k] < 1e-9*s[k], f'k={k}: modes {k} and {k+1} should coincide at Lx'
