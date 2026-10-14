"""The parametrization and its Jacobian.

Pure geometry: no eigensolve, so a failure here is unambiguous. `test_derivative.py` builds on
this and would otherwise confound a bad Jacobian with a bad shape derivative.
"""
import numpy as np
import pytest

from douse.polygon import (VertexPolygon, polygon_vertices, d_vertices_dp, param_vector,
                           perimeter_gradient, n_params, n_vertices)
from douse.polygon import polygon_area, polygon_perimeter

PERIM = 4.0        # so a regular polygon has O(1) sides and O(10) eigenvalues

# A few shapes to sweep: a square, a generic convex quadrilateral, a pentagon, and a
# deliberately lopsided quadrilateral where the closing budget is tight.
SHAPES = {
    'square':    np.array([0.0, 1.0, 1.0 + 1j, 1j]),
    'quad':      np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j]),
    'pentagon':  np.exp(2j*np.pi*np.arange(5)/5),
    'lopsided':  np.array([0.0, 2.0, 1.9 + 0.35j, 0.2 + 0.3j]),
}


def _p(name):
    return param_vector(SHAPES[name], PERIM)


@pytest.mark.parametrize('name', list(SHAPES))
def test_the_gauge_holds(name):
    """Every claim the parametrization makes about its own output."""
    p = _p(name)
    v = polygon_vertices(p, PERIM)
    assert abs(v[-1]) < 1e-14, f'last vertex must be the origin, got {v[-1]}'
    assert abs(v[0].imag) < 1e-14, f'first vertex must be real, got {v[0]}'
    assert v[0].real > 0, f'first vertex must be on the POSITIVE x axis, got {v[0]}'
    assert abs(polygon_perimeter(v) - PERIM) < 1e-12, polygon_perimeter(v)
    assert polygon_area(v) > 0, 'must be counter-clockwise'


@pytest.mark.parametrize('name', list(SHAPES))
def test_param_vector_round_trips_the_shape(name):
    """`param_vector` then `polygon_vertices` must return the SAME SHAPE -- not the same
    coordinates, since the gauge moves it. Compared by the similarity invariants."""
    v0 = SHAPES[name]
    v1 = polygon_vertices(param_vector(v0, PERIM), PERIM)
    scale = PERIM/polygon_perimeter(v0)
    assert abs(polygon_area(v1) - abs(polygon_area(v0))*scale**2) < 1e-12
    assert np.allclose(np.sort(np.abs(np.diff(np.append(v1, v1[0])))),
                       np.sort(np.abs(np.diff(np.append(v0, v0[0]))))*scale, atol=1e-12)


def test_param_vector_is_orientation_normalizing():
    """A polygon and its reversal are the same shape and must give the same p. An earlier
    version reversed on a side-length tie-break, which produced clockwise output that the
    parametrization's own `__call__` then rejected."""
    v = SHAPES['quad']
    assert np.allclose(param_vector(v, PERIM), param_vector(v[::-1], PERIM), atol=1e-13)


@pytest.mark.parametrize('name', list(SHAPES))
def test_the_jacobian_matches_a_finite_difference(name):
    """THE point of this file. Central differences on each parameter, against `d_vertices_dp`.

    The interesting row is `x_0`, which is not a parameter but is pinned by the perimeter
    constraint; every other row is an identity block or a zero, and a mistake there would be a
    bookkeeping error rather than a calculus one -- so the assertion is made on the whole matrix
    and reported per row.
    """
    p = _p(name)
    J = d_vertices_dp(p, PERIM)
    h = 1e-6
    for i in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        fd = (polygon_vertices(pp, PERIM) - polygon_vertices(pm, PERIM))/(2*h)
        got = np.concatenate([J[:len(fd), i], J[len(fd):, i]])
        want = np.concatenate([fd.real, fd.imag])
        err = np.max(np.abs(got - want))
        assert err < 1e-8, f'{name}, parameter {i}: max |analytic - FD| = {err:.2e}'


@pytest.mark.parametrize('name', list(SHAPES))
def test_the_jacobian_stays_on_the_constraint_surface(name):
    """A direct consequence worth asserting separately, because it is the property an optimizer
    depends on: every column of the Jacobian must be tangent to `perimeter = const`."""
    p = _p(name)
    v = polygon_vertices(p, PERIM)
    _, g = perimeter_gradient(v)
    grad_flat = np.concatenate([g.real, g.imag])
    drift = grad_flat @ d_vertices_dp(p, PERIM)
    assert np.max(np.abs(drift)) < 1e-12, f'perimeter drifts along p: {drift}'


def test_perimeter_gradient_matches_a_finite_difference():
    v = polygon_vertices(_p('quad'), PERIM)
    _, g = perimeter_gradient(v)
    h = 1e-7
    for k in range(len(v)):
        for axis, step in ((0, h), (1, 1j*h)):
            vp, vm = v.copy(), v.copy()
            vp[k] += step
            vm[k] -= step
            fd = (polygon_perimeter(vp) - polygon_perimeter(vm))/(2*h)
            got = g[k].real if axis == 0 else g[k].imag
            assert abs(got - fd) < 1e-7, (k, axis, got, fd)


# ── Bookkeeping and refusals ────────────────────────────────────────────────────

def test_sizes_agree():
    for n in (3, 4, 5, 8):
        assert n_params(n) == 2*n - 4
        assert n_vertices(np.zeros(n_params(n))) == n
        assert VertexPolygon(n).n_params == 2*n - 4


def test_an_impossible_perimeter_budget_is_refused():
    """The free vertices can eat the whole budget; then no polygon exists and saying so beats
    returning a complex or negative first vertex."""
    fam = VertexPolygon(4, perim=1.0)
    with pytest.raises(ValueError, match='no such polygon exists'):
        fam.vertices(np.array([0.9, 0.9, 0.0, 0.9]))
    assert not fam.is_valid(np.array([0.9, 0.9, 0.0, 0.9]))


def test_a_clockwise_p_is_refused_with_an_explanation():
    fam = VertexPolygon(4, perim=PERIM)
    p = _p('quad').copy()
    p[len(p)//2:] *= -1                     # mirror the free vertices about the x axis
    with pytest.raises(ValueError, match='clockwise'):
        fam(p)


def test_wrong_length_p_is_refused():
    with pytest.raises(ValueError, match='must have length'):
        VertexPolygon(5, perim=PERIM)(np.zeros(4))


def test_the_family_builds_a_chladni_domain():
    """chladni is douse's backend (2026-10-14): `fam(p)` is a Dirichlet chladni Polygon."""
    from chladni.geometry import Polygon
    fam = VertexPolygon(4, perim=PERIM)
    dom = fam(_p('quad'))
    assert isinstance(dom, Polygon) and dom.bc_type == "dir"
    assert abs(dom.perimeter - PERIM) < 1e-12
    assert len(dom.boundaries[0].segments) == 4


def test_the_family_still_builds_a_lappy_domain_on_request():
    """The deprecated lappy backend's domain, for `LappySpectrumEvaluator` (douse[lappy])."""
    lappy = pytest.importorskip("lappy")
    fam = VertexPolygon(4, perim=PERIM)
    dom = fam.lappy_domain(_p('quad'))
    assert isinstance(dom, lappy.Polygon)
    assert abs(dom.perimeter - PERIM) < 1e-12
    assert len(dom.bdry.segments) == 4


def test_the_geometry_helpers_match_lappys():
    """Copied from `lappy.utils` when douse stopped importing lappy for them."""
    lu = pytest.importorskip("lappy.utils")
    from douse.polygon import edge_lengths
    v = np.array([0, 1.1, 1.25 + 0.95j, 0.15 + 0.8j, -0.3 + 0.4j])
    for z in (v, v[::-1], np.c_[v.real, v.imag]):
        np.testing.assert_array_equal(edge_lengths(z), lu.edge_lengths(z))
        assert polygon_area(z) == lu.polygon_area(z)
        assert polygon_perimeter(z) == lu.polygon_perimeter(z)
