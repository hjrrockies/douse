"""The `chladni` backend: `ChladniEvaluator`, behind the same interface as `SpectrumEvaluator`.

`chladni` plans its own basis (`chladni.plan`), solves to a requested `prec` with per-level bounds
and flags, and supplies Hadamard matrices (`Spectrum.hadamard`). This checks the seam against
ground truth and against the `lappy` backend: the rectangle's closed form; agreement of both
backends on eigenvalues and on the parameter Jacobian of a generic quadrilateral; the Jacobian
against central differences of `chladni`'s own eigenvalues; and the mapping of `chladni`'s
shortfall flags onto `douse`'s refusal exceptions, so an optimizer shrinks rather than trusting a
flagged spectrum.
"""
import warnings

import numpy as np
import pytest

from douse.chladni_backend import ChladniEvaluator, refusal_for
from douse.evaluate import DegenerateSpectrum, SpectrumEvaluator, UnresolvedSpectrum, \
    SolverRefusal
from douse.polygon import VertexPolygon, param_vector

PREC = 1e-10
pytestmark = pytest.mark.filterwarnings("ignore")

QUAD = np.array([0.0, 1.1, 1.25 + 0.95j, 0.15 + 0.8j])      # a generic convex quadrilateral


def _p(vertices, perim):
    return param_vector(np.asarray(vertices, dtype=complex), perim)


def test_a_rectangles_eigenvalues_match_the_closed_form():
    from benchmarks.rectangle_exact import spectrum
    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2 * (L + H))
    ev = ChladniEvaluator(fam, n_eigs=4, prec=PREC)
    lams, _ = ev.eigenvalues(_p([0, L, L + 1j * H, 1j * H], fam.perim))
    np.testing.assert_allclose(lams, spectrum(L, H, 6)[:4], rtol=1e-9)


@pytest.mark.slow
def test_both_backends_agree_on_values_and_the_jacobian():
    fam = VertexPolygon(4, perim=float(np.sum(np.abs(np.diff(np.append(QUAD, QUAD[0]))))))
    p = _p(QUAD, fam.perim)
    lams_c, J_c = ChladniEvaluator(fam, n_eigs=3, prec=PREC)(p)
    lams_l, J_l = SpectrumEvaluator(fam, n_eigs=3)(p)
    np.testing.assert_allclose(lams_c, lams_l, rtol=1e-9)
    np.testing.assert_allclose(J_c, J_l, rtol=1e-6, atol=1e-6 * np.abs(J_l).max())


@pytest.mark.slow
def test_the_jacobian_matches_central_differences():
    fam = VertexPolygon(4, perim=float(np.sum(np.abs(np.diff(np.append(QUAD, QUAD[0]))))))
    p = _p(QUAD, fam.perim)
    ev = ChladniEvaluator(fam, n_eigs=2, prec=PREC)
    _, J = ev(p)
    # On the same evaluator, so the FROZEN plan's basis is the same at p +- h. The values carry
    # ~1e-10 relative error, so the difference carries ~lam * 1e-10 / h ~ 1e-6 relative at
    # h = 1e-4, and the truncation is h^2 ~ 1e-8. (At prec 1e-12 this shape is refused --
    # margin 1.3 -- which is the backend reporting, not failing.)
    h = 1e-4
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        fd = (ev.eigenvalues(p + e)[0] - ev.eigenvalues(p - e)[0]) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-5, atol=1e-5 * np.abs(J).max())


@pytest.mark.slow
def test_a_degenerate_level_refuses_the_simple_gradient():
    """The square: lam_2 = lam_3. douse's policy -- a confident wrong gradient is worse than a
    raise -- applies to this backend too; `cluster_matrices` is the path for a cluster."""
    fam = VertexPolygon(4, perim=4.0)
    ev = ChladniEvaluator(fam, n_eigs=3, prec=PREC)
    with pytest.raises(DegenerateSpectrum):
        ev(_p([0, 1, 1 + 1j, 1j], 4.0))


class _Spec:
    def __init__(self, flags, declined=False):
        self.flags, self.declined = tuple(flags), declined


@pytest.mark.parametrize("flags, declined, want", [
    ((), False, None),
    (("count audit: unaudited (not a prefix window)",), False, None),
    (("unmet: 2 of 4 levels miss prec=1e-10, the first at lam = 31.1 (margin 0.4; 2 is needed)",),
     False, UnresolvedSpectrum),
    (("count audit: short (mean Weyl deficit 0.9)",), False, UnresolvedSpectrum),
    (("short: found 3 of the 4 eigenvalues asked for",), False, UnresolvedSpectrum),
    (("planner: no rung met prec=1e-10; returning rung 3 of 3",), False, UnresolvedSpectrum),
    (("declined: the slope never settled",), True, SolverRefusal),
])
def test_chladni_flags_map_onto_douses_refusals(flags, declined, want):
    exc = refusal_for(_Spec(flags, declined))
    assert (exc is None) if want is None else isinstance(exc, want), exc


@pytest.mark.slow
def test_cluster_matrices_near_a_double_carry_the_members_own_gradients():
    """A nearly square rectangle: lam_2, lam_3 split slightly. Built from the members' OWN
    eigenfunctions (`lams=`), the cluster block's diagonal is each member's gradient. Checked
    against the `lappy` backend's simple gradients, an independent computation (the two backends'
    simple gradients agree to ~2e-12).

    Not checked against `lappy`'s `cluster_matrices`, which evaluates both members' Cauchy data at
    the cluster's MEAN eigenvalue: its own diagonal is off its own simple gradients by ~1e-3 here
    (measured 2026-10-12), first order in the split. `chladni` evaluates each member at its own."""
    from douse.derivative import cluster_matrices, dlams_dp
    L, H = 1.0, 1.002
    fam = VertexPolygon(4, perim=2 * (L + H))
    p = _p([0, L, L + 1j * H, 1j * H], fam.perim)
    lams_c, solver_c = ChladniEvaluator(fam, n_eigs=3, prec=PREC,
                                        degenerate_tol=0.0).eigenvalues(p)
    lams_l, solver_l = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0,
                                         warn_gap=0.0).eigenvalues(p)
    Mc = cluster_matrices(fam, p, solver_c, float(np.mean(lams_c[1:3])), 2, lams=lams_c[1:3])
    Jl = dlams_dp(fam, p, solver_l, lams_l[1:3])
    np.testing.assert_allclose(np.stack([Mc[:, 0, 0], Mc[:, 1, 1]]), Jl, rtol=1e-8,
                               atol=1e-9 * np.abs(Jl).max())
