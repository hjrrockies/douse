"""The inverse door, on the chladni backend: `SpectralMatch` through `ObjectiveFunction` and
`descend`, end to end.

The package is named for this problem and until 2026-10-14 it had never been run through an
evaluator. The target is a rectangle's closed-form spectrum, whose first three eigenvalues are
simple on a 1.2 x 0.8 rectangle (so the simple gradient applies throughout), and the family is
the perimeter-4 quadrilaterals, which contain the target up to congruence. The match is
scale-invariant, so the target's size is never used.
"""
import numpy as np
import pytest

from douse import SpectrumEvaluator, VertexPolygon, param_vector
from douse.evaluate import ObjectiveFunction
from douse.objective import SpectralMatch
from douse.optimize import descend

L, H = 1.2, 0.8
PERIM = 2 * (L + H)


def _rect_spectrum(n):
    """pi^2 (m^2/L^2 + n^2/H^2), the smallest `n` of them."""
    vals = [np.pi**2 * (m**2 / L**2 + k**2 / H**2) for m in range(1, 8) for k in range(1, 8)]
    return np.sort(vals)[:n]


def _refusing_evaluator(exc):
    from douse.errors import ShapeRefusal

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


@pytest.mark.slow
def test_a_rectangle_is_recovered_from_three_eigenvalues():
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, prec=1e-10, warn_gap=0.0)
    obj = ObjectiveFunction(ev, SpectralMatch(_rect_spectrum(3)))
    rng = np.random.default_rng(3)
    v = np.array([0, L, L + 1j * H, 1j * H]) + 0.05 * (rng.normal(size=4) + 1j * rng.normal(size=4))
    p0 = param_vector(v, PERIM)
    loss0, _ = obj(p0)
    res = descend(obj, p0, max_iter=40)
    assert res.fun < 1e-4 * loss0, (loss0, res.fun, res.message)
    edges = np.sort(np.abs(np.diff(np.append(fam.vertices(res.x), fam.vertices(res.x)[0]))))
    np.testing.assert_allclose(edges, [H, H, L, L], atol=2e-3)
    assert obj.n_refused == 0
