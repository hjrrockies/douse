"""Milestone 3: recover the known minimizers of lam_1 at fixed perimeter.

Among N-gons of given perimeter, `lam_1` is minimized by the REGULAR one. That is a theorem for
`n = 3` and `n = 4` (Polya-Szego) and open for `n >= 5`, which makes the triangle and the
quadrilateral the right acceptance tests for the whole stack: the answer is known in closed form,
so a run that lands anywhere else is a defect and not a discovery.

    equilateral triangle, perimeter 4:  lam_1 = 16 pi^2 / (3 a^2), a = 4/3  ->  3 pi^2
    square, perimeter 4:                lam_1 = 2 pi^2 / L^2,      L = 1    ->  2 pi^2

Full runs from a generic starting shape reach `-9.6e-14` and `-9.3e-13` respectively, with sides
equal to 2.5e-08 and 2.1e-07. The tests below start closer and stop sooner, because they are
regression guards and not the demonstration -- `benchmarks/known_minimizers.py` is the
demonstration.

WHAT THESE WOULD HAVE CAUGHT. The first end-to-end run converged 2.8e-03 BELOW `2 pi^2`, which is
impossible if the square is the minimizer. The cause was `track` returning a point with tension
2.0e-04 -- inside lappy's `ttol=1e-3`, so no refusal -- where the true eigenvalue was 2.9e-03
away. Without a known optimum that answer would have looked entirely reasonable. It is the reason
`SpectrumEvaluator` verifies what tracking returns, and the reason these two cases are worth their
runtime.
"""
import warnings

import numpy as np
import pytest

from douse import (VertexPolygon, param_vector, SpectrumEvaluator, ObjectiveFunction,
                   EigenvalueObjective, descend)

PERIM = 4.0


def _regularity(fam, p):
    """`(max side deviation, max angle deviation in degrees)` from the regular polygon."""
    v = fam.vertices(p)
    n = fam.n
    sides = np.abs(np.diff(np.append(v, v[0])))
    ang = np.degrees([np.pi - np.angle((v[(i + 1) % n] - v[i])/(v[i] - v[i - 1]))
                      for i in range(n)])
    return np.max(np.abs(sides - sides.mean())), np.max(np.abs(ang - 180*(n - 2)/n))


def _run(n, start, max_iter):
    fam = VertexPolygon(n, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=1)
    obj = ObjectiveFunction(ev, EigenvalueObjective(0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = descend(obj, param_vector(start, PERIM), max_iter=max_iter, gtol=1e-7)
    return fam, ev, obj, res


# ── The two theorems ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_the_triangle_recovers_the_equilateral():
    fam, ev, obj, res = _run(3, np.array([0.0, 1.5, 0.55 + 1.1j]), max_iter=60)
    exact = 3*np.pi**2
    assert abs(res.fun - exact) < 1e-9, (res.fun, exact, res.message)
    ds, da = _regularity(fam, res.x)
    assert ds < 1e-5, f'sides differ by {ds:.2e}'
    assert da < 1e-3, f'angles differ from 60 deg by {da:.2e}'


@pytest.mark.slow
def test_the_quadrilateral_recovers_the_square():
    fam, ev, obj, res = _run(4, np.array([0.0, 1.15, 1.05 + 0.95j, 0.05 + 0.9j]), max_iter=90)
    exact = 2*np.pi**2
    assert abs(res.fun - exact) < 1e-8, (res.fun, exact, res.message)
    ds, da = _regularity(fam, res.x)
    assert ds < 1e-4, f'sides differ by {ds:.2e}'
    assert da < 1e-2, f'angles differ from 90 deg by {da:.2e}'


@pytest.mark.slow
def test_the_minimum_is_approached_from_ABOVE():
    """The sharpest check available, and the one the first run failed.

    `2 pi^2` is a true minimum over this family, so no admissible shape can beat it. A value
    below it is not a better answer, it is a broken eigenvalue -- and nothing about it looks
    wrong locally. Allowed a small negative slack for the solver's own accuracy.
    """
    _, _, _, res = _run(4, np.array([0.0, 1.15, 1.05 + 0.95j, 0.05 + 0.9j]), max_iter=90)
    assert res.fun > 2*np.pi**2 - 1e-9, (
        f'converged {2*np.pi**2 - res.fun:.2e} BELOW the true minimum, which is impossible for '
        f'an admissible shape; the eigenvalue is wrong, not the optimizer')


@pytest.mark.slow
def test_tracking_carries_the_run_and_is_verified():
    """Continuation should do most of the work, and its refusals should be visible rather than
    silently accepted -- the refusal count is what tells a caller the verification is live."""
    _, ev, _, _ = _run(3, np.array([0.0, 1.5, 0.55 + 1.1j]), max_iter=60)
    assert ev.n_tracks > ev.n_solves, (ev.n_tracks, ev.n_solves)
    assert ev.n_track_refusals > 0, (
        'no tracked value was ever rejected; either the verification is not running or the '
        'threshold is too loose to be doing anything')


# ── The driver itself, with no eigensolve ───────────────────────────────────────

def _quadratic(A, b):
    """A cheap convex objective, so `descend` can be tested without a solver."""
    def f(p):
        p = np.asarray(p, dtype=float)
        return float(0.5*p @ A @ p + b @ p), A @ p + b
    return f


def test_descend_finds_the_minimum_of_a_quadratic():
    """`gtol=1e-10` and not smaller: plain descent on this quadratic reaches a gradient norm of
    ~2.5e-08 and then backtracks to nothing, so asking for 1e-12 makes it report the honest
    `no decrease` rather than the gradient test. The location is right either way, which is why
    the point is asserted separately from the stopping reason."""
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    b = np.array([-1.0, 2.0])
    exact = np.linalg.solve(A, -b)

    res = descend(_quadratic(A, b), np.array([4.0, -3.0]), is_valid=lambda p: True,
                  max_iter=500, gtol=1e-7)
    assert res.success, res.message
    assert np.allclose(res.x, exact, atol=1e-7), res.x

    tight = descend(_quadratic(A, b), np.array([4.0, -3.0]), is_valid=lambda p: True,
                    max_iter=500, gtol=1e-14)
    assert not tight.success and 'no decrease' in tight.message
    assert np.allclose(tight.x, exact, atol=1e-9), tight.x


def test_descend_respects_the_admissible_set():
    """An inadmissible trial point must never reach the objective -- `is_valid` is cheap and the
    objective is not."""
    seen = []

    def f(p):
        seen.append(np.array(p))
        return float(p @ p), 2*np.array(p, dtype=float)

    res = descend(f, np.array([1.0, 1.0]), is_valid=lambda p: np.all(np.asarray(p) > 0.3),
                  max_iter=50, gtol=1e-10)
    assert all(np.all(p > 0.3) for p in seen), 'the objective saw an inadmissible point'
    assert np.all(res.x > 0.3)


def test_descend_refuses_an_inadmissible_start():
    with pytest.raises(ValueError, match='not an admissible parameter vector'):
        descend(_quadratic(np.eye(2), np.zeros(2)), np.array([1.0, 1.0]),
                is_valid=lambda p: False)


def test_descend_reports_stopping_honestly():
    """`success` is for the gradient test alone. Running out of iterations is reported as a
    failure with a message saying so, not dressed up as convergence."""
    res = descend(_quadratic(np.eye(2), np.array([1.0, 1.0])), np.array([5.0, 5.0]),
                  is_valid=lambda p: True, max_iter=2, gtol=1e-14)
    assert not res.success
    assert 'max_iter' in res.message
    assert res.nit == 2 and res.nfev >= 2
