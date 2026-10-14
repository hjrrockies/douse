"""The inverse door, on the chladni backend: `SpectralMatch` through `gauss_newton`, end to end.

The package is named for this problem and until 2026-10-14 it had never been run through an
evaluator. The target is a rectangle's closed-form spectrum, whose first six eigenvalues are
simple on a 1.2 x 0.8 rectangle, and the family is the perimeter-4 quadrilaterals, which contain
the target up to congruence. The match is scale-invariant, so the target's size is never used --
and that is why SIX eigenvalues: `K - 1` ratios must cover the family's `2N - 4 = 4` parameters.
The first version of this test used three, whose two ratios leave a two-dimensional manifold of
"solutions", and read a slow descent along it as a failure of the optimizer.
"""
import numpy as np
import pytest

from douse import SpectrumEvaluator, VertexPolygon, param_vector
from douse.evaluate import ObjectiveFunction
from douse.objective import SpectralMatch
from douse.optimize import gauss_newton

L, H = 1.2, 0.8
PERIM = 2 * (L + H)


def _rect_spectrum(n):
    """pi^2 (m^2/L^2 + k^2/H^2), the smallest `n` of them."""
    vals = [np.pi**2 * (m**2 / L**2 + k**2 / H**2) for m in range(1, 8) for k in range(1, 8)]
    return np.sort(vals)[:n]


def _refusing_evaluator(exc):
    class Fam:
        def is_valid(self, p):
            return True

    class Ev:
        family, n_eigs = Fam(), 3

        def __call__(self, p):
            raise exc
    return Ev()


def test_a_refusal_is_an_invalid_value_not_a_dead_run():
    """`descend` backs off from `inf` as it does for an inadmissible `p`; the refusal is counted
    and its message kept in the history."""
    from douse.errors import UnresolvedSpectrum
    obj = ObjectiveFunction(_refusing_evaluator(UnresolvedSpectrum("chladni: unmet")),
                            SpectralMatch(_rect_spectrum(3)))
    val, grad = obj(np.zeros(4))
    assert val == np.inf and not grad.any()
    assert obj.n_refused == 1 and obj.history[-1]['refused'].startswith("chladni")


def test_the_residual_is_consistent_with_the_loss_and_its_gradient():
    """`loss == r.r` and `grad == 2 J'r`, on a synthetic spectrum and Jacobian."""
    rng = np.random.default_rng(0)
    match = SpectralMatch(_rect_spectrum(6))
    lams = _rect_spectrum(6) * (1 + 0.05 * rng.normal(size=6))
    J = rng.normal(size=(6, 4))
    loss, grad = match(lams, J)
    r, Jr = match.residual(lams, J)
    assert loss == pytest.approx(r @ r, rel=1e-12)
    np.testing.assert_allclose(grad, 2 * Jr.T @ r, rtol=1e-12)


def test_gauss_newton_on_a_consistent_linear_residual():
    """A linear residual is solved in one step whatever the damping does afterwards."""
    rng = np.random.default_rng(1)
    A, p_true = rng.normal(size=(6, 4)), rng.normal(size=4)

    class Match:
        n_eigs = 7

        def residual(self, lams, dlams):
            return A @ (lams[:4] - p_true), A @ dlams[:4]

    class Ev:
        n_eigs = 7

        class family:
            @staticmethod
            def is_valid(p):
                return True

        def __call__(self, p):
            return np.r_[p, 0, 0, 0], np.vstack([np.eye(4), np.zeros((3, 4))])

    res = gauss_newton(Ev(), Match(), np.zeros(4), max_step=10.0)
    assert res.success and np.allclose(res.x, p_true, atol=1e-10)


@pytest.mark.slow
def test_a_rectangle_is_recovered_from_six_eigenvalues():
    """Recovery is SECOND order at a rectangle, and the tolerances say so. A shear of the
    rectangle into a parallelogram by `eps` and by `-eps` are mirror images, so every eigenvalue
    is even in `eps`: the shear is a null direction of the ratio Jacobian at the solution, and a
    residual of `r` leaves an edge error of order `sqrt(r)`. Measured 2026-10-14: the default
    `rtol = 1e-14` stopped at residual 5e-8 with the edges 4e-4 off; the solver's own floor
    (ratios to ~1e-10) bounds what any tolerance can reach at ~1e-5."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=6, prec=1e-10, warn_gap=0.0)
    match = SpectralMatch(_rect_spectrum(6))
    rng = np.random.default_rng(3)
    v = np.array([0, L, L + 1j * H, 1j * H]) + 0.05 * (rng.normal(size=4) + 1j * rng.normal(size=4))
    p0 = param_vector(v, PERIM)
    res = gauss_newton(ev, match, p0, max_iter=40, rtol=1e-19)
    assert res.fun < 1e-15, (res.fun, res.nit, res.message)
    z = fam.vertices(res.x)
    edges = np.sort(np.abs(np.diff(np.append(z, z[0]))))
    np.testing.assert_allclose(edges, [H, H, L, L], atol=1e-4)
    angles = np.degrees(np.abs(np.angle(-np.roll(np.diff(np.append(z, z[0])), 1)
                                         / np.diff(np.append(z, z[0])))))
    np.testing.assert_allclose(angles, 90.0, atol=0.02)
    assert res.n_refused == 0
