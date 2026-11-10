"""`FourierPolygon`: the chart, its Jacobian, its symmetries, and the spectra it produces.

The go/no-go smoke tests of `docs/todo.md`'s FourierPolygon item (adopted 2026-11-10): if any of
these fail, the comparison runs do not happen and `VertexPolygon` stays.
"""
import numpy as np
import pytest

from douse import FourierPolygon
from douse.constraints import Convexity, MinEdgeLength
from douse.polygon import polygon_area, polygon_perimeter, signed_frequency

PERIM = 4.0
SIZES = [3, 4, 5, 6, 8]


def _fd(f, p, h=1e-6):
    """Five-point central difference Jacobian of a vector-valued `f`."""
    p = np.asarray(p, dtype=float)
    cols = []
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        cols.append((f(p - 2*e) - 8*f(p - e) + 8*f(p + e) - f(p + 2*e))/(12*h))
    return np.array(cols).T


def _xy(v):
    return np.concatenate([v.real, v.imag])


def _generic(fam, scale=0.08, seed=0):
    return scale*np.random.default_rng(seed + fam.n).normal(size=fam.n_params)


def test_signed_frequency():
    assert [signed_frequency(k, 6) for k in range(6)] == [0, 1, 2, 3, -2, -1]
    assert [signed_frequency(k, 5) for k in range(5)] == [0, 1, 2, -2, -1]


@pytest.mark.parametrize('n', SIZES)
def test_p_zero_is_the_regular_polygon(n):
    fam = FourierPolygon(n, PERIM)
    v = fam.vertices(np.zeros(fam.n_params))
    assert np.allclose(np.abs(v), np.abs(v[0]), atol=1e-14)
    assert np.allclose(np.abs(np.diff(np.append(v, v[0]))), PERIM/n, atol=1e-14)
    assert abs(v.mean()) < 1e-14
    assert fam.x1_share(np.zeros(fam.n_params)) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize('s', [0, 1])
@pytest.mark.parametrize('n', SIZES)
def test_the_chart_round_trips_and_keeps_the_perimeter(n, s):
    fam = FourierPolygon(n, PERIM, s)
    p = _generic(fam)
    v = fam.vertices(p)
    assert abs(polygon_perimeter(v) - PERIM) < 1e-13
    assert polygon_area(v) > 0
    assert np.allclose(fam.from_vertices(v), p, atol=1e-13)
    assert np.allclose(fam.from_vertices(3.7*v + (2 - 1j)), p, atol=1e-13)     # similarity-blind


@pytest.mark.parametrize('s', [0, 1])
@pytest.mark.parametrize('n', SIZES)
def test_the_jacobian_matches_a_finite_difference(n, s):
    fam = FourierPolygon(n, PERIM, s)
    p = _generic(fam)
    assert np.abs(fam.d_vertices_dp(p) - _fd(lambda q: _xy(fam.vertices(q)), p)).max() < 1e-9


@pytest.mark.parametrize('n', SIZES)
def test_the_plain_chart_is_an_isometry_at_the_regular_polygon(n):
    """So a trust radius in `p` means what it means for `VertexPolygon` (`delta0`, the replan
    radius and the rest of `cluster_opt`'s constants are in those units)."""
    fam = FourierPolygon(n, PERIM, 0)
    J = fam.d_vertices_dp(np.zeros(fam.n_params))
    assert np.allclose(J.T @ J, np.eye(fam.n_params), atol=1e-14)


@pytest.mark.parametrize('n', [5, 6, 8])
def test_the_sobolev_weight_makes_high_modes_expensive(n):
    """`J^T J` at the regular polygon is `diag(1/w_k)`: the affine mode costs 1, zigzags most."""
    fam = FourierPolygon(n, PERIM, 1)
    m = n - 2
    d = np.diag(fam.d_vertices_dp(np.zeros(fam.n_params)).T @ fam.d_vertices_dp(np.zeros(fam.n_params)))
    expect = np.array([1/signed_frequency(k, n)**2 for k in range(2, n)])
    assert np.allclose(d[:m], expect, atol=1e-14) and np.allclose(d[m:], expect, atol=1e-14)
    assert d[m - 1] == pytest.approx(1.0)                                         # mode n-1


@pytest.mark.parametrize('s', [0, 1])
@pytest.mark.parametrize('n', [5, 6, 8])
def test_relabelling_is_a_phase_and_reflection_keeps_the_moduli(n, s):
    """No vertex is special: relabelling multiplies `z_k` by `w^((k-1)r)`, and the mirror image
    has the same `|z_k|`, so both keep `||p||` -- the trust region sees one shape, not 2n."""
    fam = FourierPolygon(n, PERIM, s)
    p = _generic(fam)
    v, z = fam.vertices(p), fam.z(p)
    w = np.exp(2j*np.pi/n)
    for r in range(1, n):
        q = fam.from_vertices(np.roll(v, -r))
        assert np.allclose(fam.z(q), z*w**((fam.modes - 1)*r), atol=1e-13)
        assert np.linalg.norm(q) == pytest.approx(np.linalg.norm(p), abs=1e-13)
    q = fam.from_vertices(np.conj(v))
    assert np.allclose(np.abs(fam.z(q)), np.abs(z), atol=1e-13)


@pytest.mark.parametrize('n', SIZES)
def test_the_signed_area_formula(n):
    """`A = (n/2) sum_k sin(2 pi k/n) |X_k|^2` over the vertices' DFT."""
    fam = FourierPolygon(n, PERIM, 0)
    v = fam.vertices(_generic(fam))
    X = np.fft.fft(v)/n
    A = 0.5*n*np.sum(np.sin(2*np.pi*np.arange(n)/n)*np.abs(X)**2)
    assert A == pytest.approx(polygon_area(v), abs=1e-14)


@pytest.mark.parametrize('constraint', [MinEdgeLength, Convexity])
@pytest.mark.parametrize('s', [0, 1])
def test_vertex_constraints_chain_through_the_chart(constraint, s):
    fam = FourierPolygon(6, PERIM, s)
    con = constraint(fam)
    p = _generic(fam)
    _, J = con(p)
    assert np.abs(J - _fd(lambda q: con(q)[0], p)).max() < 1e-8


def test_refusals():
    fam = FourierPolygon(5, PERIM)
    with pytest.raises(ValueError, match='length'):
        fam.vertices(np.zeros(5))
    star = np.exp(4j*np.pi*np.arange(5)/5)             # the pentagram: X_1 = 0, the chart's hole
    with pytest.raises(ValueError, match='edge of the Fourier chart'):
        fam.from_vertices(star)
    m = fam.n - 2
    p = np.zeros(fam.n_params)
    p[m - 1] = 2.0*fam._scale[m - 1]                   # z_{n-1} = 2: u_j = w^j + 2 w^-j, clockwise
    assert polygon_area(fam.vertices(p)) < 0
    assert not fam.is_valid(p)
    with pytest.raises(ValueError, match='clockwise'):
        fam(p)


def _spectrum(fam, p, n_eigs, prec):
    from douse.chladni_backend import SpectrumEvaluator
    ev = SpectrumEvaluator(fam, n_eigs=n_eigs, prec=prec, degenerate_tol=0.0, warn_gap=0.0)
    return np.asarray(ev.eigenvalues(p)[0], dtype=float)


def test_the_regular_pentagon_has_the_known_ratio():
    """`lam_2/lam_1 = 2.5268387` at the regular pentagon (`docs/todo.md`, the scope table)."""
    fam = FourierPolygon(5, PERIM)
    lam = _spectrum(fam, np.zeros(fam.n_params), 3, 1e-10)
    assert lam[1]/lam[0] == pytest.approx(2.5268387, abs=5e-8)


@pytest.mark.parametrize('s', [0, 1])
def test_a_rectangle_in_the_chart_has_the_closed_form_spectrum(s):
    """A rectangle is modes 1 and 3 only (`z_2 = 0`); its spectrum is `pi^2(m^2/L^2 + n^2/H^2)`."""
    L, H = 1.2, 0.8
    fam = FourierPolygon(4, 2*(L + H), s)
    p = fam.from_vertices(np.array([0, L, L + 1j*H, 1j*H]))
    assert abs(fam.z(p)[0]) < 1e-15
    exact = np.sort([np.pi**2*(m*m/L**2 + k*k/H**2) for m in range(1, 5) for k in range(1, 5)])
    lam = _spectrum(fam, p, 4, 1e-10)
    assert np.allclose(lam, exact[:4], rtol=1e-9)


def test_runs_builds_either_chart_with_its_own_constraints():
    from douse import runs
    from douse.constraints import ClosingBudget
    v, f = runs.family_for(6), runs.family_for(6, chart='fourier', s=1)
    assert type(v).__name__ == 'VertexPolygon' and isinstance(f, FourierPolygon) and f.s == 1
    kinds = lambda fam: [type(c) for c in runs.constraints_for(fam).constraints]
    assert ClosingBudget in kinds(v) and ClosingBudget not in kinds(f)
    with pytest.raises(ValueError):
        runs.family_for(6, chart='nope')


@pytest.mark.slow
@pytest.mark.parametrize('s', [0, 1])
def test_dlam_dp_through_the_chart_matches_a_finite_difference(s):
    """The shape derivative chained through the chart, against five-point differences of cold
    solves in `p` itself (so along the fixed-perimeter surface the optimizer walks)."""
    from douse.chladni_backend import SpectrumEvaluator
    from douse.derivative import dlams_dp
    fam = FourierPolygon(4, PERIM, s)
    p0 = _generic(fam, 0.1)

    def lam1(p):
        ev = SpectrumEvaluator(fam, n_eigs=1, prec=1e-11, degenerate_tol=0.0, warn_gap=0.0)
        return ev.eigenvalues(p)

    lams, solver = lam1(p0)
    g = dlams_dp(fam, p0, solver, lams[:1])[0]
    h = 1e-3
    fd = np.array([(lam1(p0 - 2*h*e)[0][0] - 8*lam1(p0 - h*e)[0][0] + 8*lam1(p0 + h*e)[0][0]
                    - lam1(p0 + 2*h*e)[0][0])/(12*h) for e in np.eye(fam.n_params)])
    assert np.abs(g - fd).max() < 1e-6, (g, fd)


@pytest.mark.slow
@pytest.mark.parametrize('s', [0, 1])
def test_the_square_is_reached_through_the_degeneracy_in_the_fourier_chart(s):
    """`test_cluster_opt.py`'s acceptance test, same start shape, in this chart."""
    import warnings
    from douse import SpectrumEvaluator
    from douse.cluster_opt import maximize_ratio
    fam = FourierPolygon(4, PERIM, s)
    rng = np.random.default_rng(0)
    v = PERIM/4/(2*np.sin(np.pi/4))*np.exp(2j*np.pi*np.arange(4)/4)
    v = v + 0.06*(rng.normal(size=4) + 1j*rng.normal(size=4))*np.abs(v).mean()
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, fam.from_vertices(v), k=1, max_iter=120)
    assert abs(res.fun - 2.5) < 1e-6, (res.fun, res.message)
    c = res.final_case
    assert c['row'] == 'lowest_2' and c['certified'] and c['at_endpoint'], c
