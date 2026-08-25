"""Geometric constraints, and degeneration as a result rather than a failure.

THE CLAIM THIS FILE EXISTS TO PIN. A minimum edge length is a GAUGE FIXING, not a restriction on
which domains the optimizer can reach. An N-gon loses a vertex either by collapsing an edge or by
flattening one, and both reach the same limit domain -- so forbidding the first leaves every
M-gon still reachable via the second, with every edge comfortably above the floor.
`test_every_m_gon_is_still_reachable_under_the_floor` is that argument as a test; if it ever
fails, the constraint has stopped being a gauge choice and started deciding the answer.

The Jacobians are all differenced. A wrong constraint gradient does not crash -- it quietly bends
the feasible direction, which is the same failure mode as a wrong shape derivative.
"""
import numpy as np
import pytest

from douse import VertexPolygon, param_vector, polygon_vertices
from douse import degeneracy as dg
from douse.constraints import ClosingBudget, ConstraintSet, MinEdgeLength
from douse.polygon import d_vertices_dp, gauge_vector

PERIM = 4.0


def _fd(f, p, h=1e-6):
    """Five-point central difference Jacobian of a vector-valued `f`."""
    p = np.asarray(p, dtype=float)
    cols = []
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        cols.append((f(p - 2*e) - 8*f(p - e) + 8*f(p + e) - f(p + 2*e))/(12*h))
    return np.array(cols).T


def _regular(n, perim=PERIM):
    return perim/n/(2*np.sin(np.pi/n))*np.exp(2j*np.pi*np.arange(n)/n)


def _square_with_flats(n_extra, perim=PERIM):
    """The unit square carrying `n_extra` flat vertices, scaled to `perim`."""
    corners = np.array([0, 1, 1+1j, 1j], dtype=complex)
    v = list(corners)
    for k in range(n_extra):
        i = k % 4
        v.insert(2*i + 1, corners[i] + 0.5*(corners[(i+1) % 4] - corners[i]))
    v = np.array(v, dtype=complex)
    return v*perim/np.abs(np.diff(np.r_[v, v[0]])).sum()


# ---------------------------------------------------------------- values and Jacobians

@pytest.mark.parametrize('n', [4, 5, 6])
def test_min_edge_length_value_is_the_edge_lengths(n):
    fam = VertexPolygon(n, perim=PERIM)
    p = param_vector(_regular(n)*1.0 + 0.05*np.arange(n), PERIM)
    con = MinEdgeLength(fam, l_min=0.05)
    v = fam.vertices(p)
    expected = np.abs(v - np.roll(v, 1)) - 0.05
    assert np.allclose(con.value(p), expected, atol=1e-14)


@pytest.mark.parametrize('n', [4, 5, 6])
def test_min_edge_length_jacobian_matches_a_finite_difference(n):
    fam = VertexPolygon(n, perim=PERIM)
    rng = np.random.default_rng(n)
    p = param_vector(_regular(n) + 0.08*(rng.normal(size=n) + 1j*rng.normal(size=n)), PERIM)
    con = MinEdgeLength(fam)
    _, J = con(p)
    assert np.allclose(J, _fd(con.value, p), rtol=1e-6, atol=1e-8), np.abs(J - _fd(con.value, p)).max()


@pytest.mark.parametrize('n', [4, 5, 6])
def test_closing_budget_jacobian_matches_a_finite_difference(n):
    fam = VertexPolygon(n, perim=PERIM)
    rng = np.random.default_rng(10 + n)
    p = param_vector(_regular(n) + 0.05*(rng.normal(size=n) + 1j*rng.normal(size=n)), PERIM)
    con = ClosingBudget(fam)
    _, J = con(p)
    assert np.allclose(J, _fd(con.value, p), rtol=1e-6, atol=1e-8)


def test_a_constraint_set_stacks_and_reports_its_worst():
    fam = VertexPolygon(5, perim=PERIM)
    p = param_vector(_regular(5), PERIM)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    c, J = cons(p)
    assert c.shape == (7,) and J.shape == (7, fam.n_params)
    assert cons.is_feasible(p)
    i, worst = cons.worst(p)
    assert worst > 0 and 0 <= i < 7


def test_an_empty_constraint_set_is_shaped_correctly():
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(_regular(4), PERIM)
    c, J = ConstraintSet()(p)
    assert c.shape == (0,) and J.shape == (0, fam.n_params)


# ---------------------------------------------------------------- the gauge argument

@pytest.mark.parametrize('n_extra', [1, 2, 3, 4])
def test_every_m_gon_is_still_reachable_under_the_floor(n_extra):
    """THE LOAD-BEARING TEST, as a statement about GEOMETRY. A square written as a (4+k)-gon with
    k flat vertices must clear the default edge floor with room to spare -- otherwise the floor
    is not a gauge fixing, it is a restriction, and the justification for imposing it collapses.

    Deliberately stated on the vertices rather than on `p`: the claim is that the DOMAIN is still
    reachable, which is independent of how the parametrization happens to label it.
    """
    n = 4 + n_extra
    v = _square_with_flats(n_extra)
    assert len(v) == n
    l_min = MinEdgeLength(VertexPolygon(n, perim=PERIM)).l_min

    assert dg.effective_vertices(v) == 4, 'it should still be a square'
    lengths = np.abs(v - np.roll(v, 1))
    assert np.all(lengths > l_min), (
        f'the {n}-gon representation of the square violates l_min={l_min:g}: {lengths}')
    assert lengths.min() > 3*l_min, 'and with room to spare, not marginally'


@pytest.mark.parametrize('n_extra', [1, 2, 3, 4])
def test_the_flat_representation_survives_the_round_trip_through_p(n_extra):
    """...and the parametrization must be able to HOLD it, which is a separate claim and one
    that failed when first tested.

    `param_vector` puts the longest side last, and on the 8-gon square every side ties at 0.5, so
    it rolls a FLAT vertex into `v_0`. `v_0` is the vertex `polygon_vertices` solves for, and a
    flat one makes the two closing edges collinear -- the closing budget `C` then equals `|v_1|`
    exactly and the constructor refuses with "no such polygon exists", on a perfectly good shape.
    `canonicalize_p` goes through `gauge_vector` and anchors a genuine corner instead.
    """
    n = 4 + n_extra
    v = _square_with_flats(n_extra)
    p = gauge_vector(dg.canonicalize(v), PERIM, anchor=0)
    back = polygon_vertices(p, PERIM)
    assert dg.effective_vertices(back) == 4
    assert abs(dg.turning_angles(back)[0]) > 1e-9, 'the anchor must be a genuine corner'
    assert np.all(np.isfinite(d_vertices_dp(p, PERIM))), 'and the Jacobian must exist'
    con = MinEdgeLength(VertexPolygon(n, perim=PERIM))
    assert np.all(con.value(p) > 0)


def test_the_floor_is_well_below_the_average_edge():
    for n in (3, 4, 5, 8, 12):
        con = MinEdgeLength(VertexPolygon(n, perim=PERIM))
        assert con.l_min == pytest.approx(PERIM/(10*n))
        assert con.l_min < 0.2*PERIM/n


def test_an_impossible_floor_is_refused():
    fam = VertexPolygon(4, perim=PERIM)
    with pytest.raises(ValueError, match='no polygon satisfies'):
        MinEdgeLength(fam, l_min=PERIM/3)


# ---------------------------------------------------------------- degeneracy diagnostics

def test_turning_angles_sum_to_two_pi_and_vanish_at_a_flat_vertex():
    v = _square_with_flats(2)
    turn = dg.turning_angles(v)
    assert abs(turn.sum() - 2*np.pi) < 1e-12
    assert np.sum(np.abs(turn) < 1e-12) == 2
    assert np.allclose(np.degrees(dg.interior_angles(v))[np.abs(turn) < 1e-12], 180.0)


def test_the_report_separates_flattening_from_a_collapsed_edge():
    """The two degenerations need different handling, so they must not read the same. Flattening
    is exact and canonicalizable; a collapsed edge means the number needs re-solving."""
    flat = dg.report(_square_with_flats(1))
    assert flat.degenerate and not flat.collapsed_edge
    assert flat.n_effective == 4 and 'flattening' in flat.verdict

    short = np.array([0, 1, 1 + 1e-9j, 1+1j, 1j], dtype=complex)
    rep = dg.report(short*PERIM/np.abs(np.diff(np.r_[short, short[0]])).sum())
    assert rep.collapsed_edge and 're-solving' in rep.verdict

    clean = dg.report(_regular(5))
    assert not clean.degenerate and clean.n_effective == 5


# ---------------------------------------------------------------- canonicalization

@pytest.mark.parametrize('n_extra', [1, 2, 3])
def test_canonicalize_preserves_the_domain(n_extra):
    """It is only allowed because it is exact: same corners, same area, same perimeter."""
    from lappy.utils import polygon_area, polygon_perimeter
    v = _square_with_flats(n_extra)
    out = dg.canonicalize(v)
    assert len(out) == len(v)
    assert dg.effective_vertices(out) == 4
    assert abs(polygon_area(out) - polygon_area(v)) < 1e-12
    assert abs(polygon_perimeter(out) - polygon_perimeter(v)) < 1e-12

    corners = lambda z: np.sort_complex(np.round(z[np.abs(dg.turning_angles(z)) > 1e-9], 9))
    assert np.allclose(corners(out), corners(v)), (corners(out), corners(v))


def test_canonicalize_spreads_flat_vertices_that_were_piled_up():
    """Sliding a flat vertex is free, so a run can park several in one place and reintroduce the
    short edges the floor exists to prevent. They must come back out."""
    piled = np.array([0, 0.005, 0.01, 1, 1+1j, 1j], dtype=complex)
    piled = piled*PERIM/np.abs(np.diff(np.r_[piled, piled[0]])).sum()
    assert dg.report(piled).min_edge < 1e-2

    out = dg.canonicalize(piled)
    assert dg.effective_vertices(out) == 4
    assert dg.report(out).min_edge > MinEdgeLength(VertexPolygon(6, perim=PERIM)).l_min


def test_canonicalize_moves_the_anchor_off_a_flat_vertex():
    """`d_vertices_dp` divides by `dP/dx_0`, which is exactly zero when `v_0` is flat -- so a run
    that flattens its anchor loses every gradient. Relabelling is a rotation of the index: same
    shape, different `p`, and the Jacobian comes back."""
    v = np.array([0.5, 1, 1+1j, 1j, 0], dtype=complex)      # v_0 is flat, mid-bottom-edge
    v = v*PERIM/np.abs(np.diff(np.r_[v, v[0]])).sum()
    assert abs(dg.turning_angles(v)[0]) < 1e-12

    p_bad = param_vector(v, PERIM)
    bad_v = polygon_vertices(p_bad, PERIM)
    if abs(dg.turning_angles(bad_v)[0]) < 1e-9:
        with pytest.raises(ValueError, match='dP/dx_0 vanishes'):
            d_vertices_dp(p_bad, PERIM)

    p_ok = dg.canonicalize_p(p_bad, PERIM)
    assert abs(dg.turning_angles(polygon_vertices(p_ok, PERIM))[0]) > 1e-9
    J = d_vertices_dp(p_ok, PERIM)
    assert np.all(np.isfinite(J))


def test_canonicalize_is_a_no_op_on_a_non_degenerate_polygon():
    v = _regular(5)
    assert np.allclose(dg.canonicalize(v), v)


def test_canonicalize_refuses_a_shape_that_is_no_longer_a_polygon():
    almost = np.array([0, 0.5, 1, 1.5], dtype=complex)      # collinear: no genuine corners
    with pytest.raises(ValueError, match='not a polygon'):
        dg.canonicalize(almost)
