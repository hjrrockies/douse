"""Cluster machinery, the ratio objective, and what can be said at a degenerate optimum.

The target problem -- maximize `lam_k/lam_1` over N-gons, the polygon analogue of Osting's
Fourier-boundary work -- has its conjectured optimum ON a degeneracy: at every regular N-gon
measured, `lam_2 = lam_3` exactly. That is where a gradient method stops converging, so what is
tested here is the alternative: read off the cluster structure directly.

THE HEADLINE, AND IT IS A NEGATIVE ONE. The first-order optimality condition is satisfied
AUTOMATICALLY at these points, not as evidence. `tr M = 0` forces `C = -A`, hence
`det M(d) = -(A.d)^2 - (B.d)^2 <= 0` identically and `Q = -A A^T - B B^T` has rank at most 2
whatever `n_p` is. An earlier version of `certify.py` reported that as a certificate; it is not
one, and these tests pin the algebra so it cannot be re-mistaken.
"""
import warnings

import numpy as np
import pytest

lappy = pytest.importorskip("lappy")   # these drive lappy solvers directly: the deprecated backend
from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse import VertexPolygon, param_vector
from douse.derivative import cluster_matrices, detect_cluster, directional_derivatives
from douse.objective import RatioObjective
from douse.certify import det_form, first_order_report, second_order_probe

PERIM = 4.0


def _regular(N, perim=PERIM):
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


# ── Arithmetic, no eigensolve ───────────────────────────────────────────────────

def test_detect_cluster_groups_by_relative_gap():
    lams = np.array([1.0, 5.0, 5.0 + 1e-12, 9.0, 9.0 + 1e-11, 9.0 + 2e-11])
    assert detect_cluster(lams, 0) == (0, 1)
    assert detect_cluster(lams, 1) == (1, 3)
    assert detect_cluster(lams, 2) == (1, 3)
    assert detect_cluster(lams, 3) == (3, 6)
    # A loose enough tolerance swallows the whole spectrum.
    assert detect_cluster(lams, 1, rtol=1.0) == (0, 6)

    # ...and at intermediate tolerances membership is NOT symmetric in the starting index,
    # because the gap is compared against the value at the current endpoint. Growing down from
    # index 3, the 9.0 -> 5.0 gap of 4.0 clears rtol * 9.0 = 4.5; growing up from index 1 the
    # same gap must clear rtol * 5.0 = 2.5 and does not. Worth pinning: it means "the cluster"
    # is only well defined when the tolerance is far below every real gap, which is how the
    # certify path uses it.
    assert detect_cluster(lams, 1, rtol=0.5) == (1, 3)
    assert detect_cluster(lams, 3, rtol=0.5) == (1, 6)
    with pytest.raises(IndexError):
        detect_cluster(lams, 99)


def test_det_form_reproduces_the_determinant():
    """`det(sum_i d_i M_i)` is a quadratic form in `d` for 2x2 blocks -- the fact the whole
    first-order analysis rests on."""
    rng = np.random.default_rng(0)
    Ms = rng.normal(size=(5, 2, 2))
    Ms = 0.5*(Ms + np.transpose(Ms, (0, 2, 1)))
    Q = det_form(Ms)
    for _ in range(50):
        d = rng.normal(size=5)
        assert abs(d @ Q @ d - np.linalg.det(np.einsum('i,ijk->jk', d, Ms))) < 1e-10


def test_a_traceless_cluster_forces_a_negative_semidefinite_form():
    """The reason first order settles nothing. `tr M = 0` means `C = -A`, so
    `det M(d) = -(A.d)^2 - (B.d)^2` and `Q = -A A^T - B B^T`: negative semidefinite by
    construction, rank at most 2 however many parameters there are."""
    rng = np.random.default_rng(1)
    n_p = 7
    A, B = rng.normal(size=n_p), rng.normal(size=n_p)
    Ms = np.array([[[A[i], B[i]], [B[i], -A[i]]] for i in range(n_p)])
    assert np.allclose(np.trace(Ms, axis1=1, axis2=2), 0)
    Q = det_form(Ms)
    w = np.linalg.eigvalsh(Q)
    assert w.max() < 1e-10*abs(w).max(), w
    assert np.sum(np.abs(w) > 1e-8*abs(w).max()) == 2, f'rank should be 2, got {w}'
    assert np.allclose(Q, -np.outer(A, A) - np.outer(B, B))


def test_directional_derivatives_are_linear_in_the_direction():
    rng = np.random.default_rng(2)
    Ms = rng.normal(size=(3, 2, 2))
    Ms = 0.5*(Ms + np.transpose(Ms, (0, 2, 1)))
    d = rng.normal(size=3)
    assert np.allclose(directional_derivatives(Ms, 2.5*d), 2.5*directional_derivatives(Ms, d))
    assert np.allclose(directional_derivatives(Ms, -d),
                       -directional_derivatives(Ms, d)[::-1])


def test_ratio_objective_gradient():
    lams = np.array([9.87, 24.67, 24.68, 39.5])
    rng = np.random.default_rng(3)
    dl = rng.normal(size=(4, 5))
    obj = RatioObjective(1, sign=1.0)
    _, g = obj(lams, dl)
    h = 1e-7
    fd = np.array([((lams[1] + h*dl[1, i])/(lams[0] + h*dl[0, i])
                    - (lams[1] - h*dl[1, i])/(lams[0] - h*dl[0, i]))/(2*h) for i in range(5)])
    assert np.max(np.abs(g - fd)) < 1e-7


def test_ratio_objective_is_scale_invariant():
    """Scaling a domain divides every eigenvalue by the same factor, so the ratio must not move.
    This is why the fixed perimeter is pure gauge for this objective."""
    lams = np.array([9.87, 24.67, 39.5])
    obj = RatioObjective(1, sign=1.0)
    base, _ = obj(lams, np.zeros((3, 4)))
    for c in (0.3, 7.0):
        val, _ = obj(lams/c**2, np.zeros((3, 4)))
        assert abs(val - base) < 1e-14


def test_ratio_objective_rejects_k_zero():
    with pytest.raises(ValueError, match='lam_1/lam_1 is constant'):
        RatioObjective(0)


# ── At a real degeneracy ────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def square():
    """The regular 4-gon, where lam_2 = lam_3 exactly."""
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(_regular(4), PERIM)
    dom = fam.lappy_domain(p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, weyl_est(6, dom), target=1e-13)
        solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom),
                                            basis=BP.realize(plan, dom), prec=1e-12)
        lams = np.asarray(Eigenproblem(dom, eval_solver=solver, precision=1e-12).solve(3),
                          dtype=float)

        def ratio(q):
            d = fam.lappy_domain(q)
            s = MPSEigensolver.from_domain(d, lam_max=weyl_est(6, d),
                                           basis=BP.realize(plan, d), prec=1e-12)
            e = np.asarray(Eigenproblem(d, eval_solver=s, precision=1e-12).solve(2), dtype=float)
            return e[1]/e[0]

    return fam, p, solver, lams, ratio


@pytest.mark.slow
def test_the_square_carries_an_exact_double(square):
    fam, p, solver, lams, _ = square
    assert abs(lams[1] - 5*np.pi**2) < 1e-9, lams
    assert abs(lams[2] - lams[1]) < 1e-9*lams[1], f'lam_2 and lam_3 should coincide: {lams}'
    assert detect_cluster(lams, 1) == (1, 3)


@pytest.mark.slow
def test_the_structural_facts_hold_at_the_square(square):
    """Both hypotheses the first-order reduction needs, measured rather than assumed.

    `grad lam_1 = 0` because the square MINIMIZES lam_1 at fixed perimeter -- this package's own
    Milestone 3 result. `tr M = 0` because lam_2 + lam_3 is stationary there.
    """
    fam, p, solver, lams, _ = square
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rep = first_order_report(fam, p, solver, lams, k=1)
    h = rep['hypotheses']
    assert h['grad_lam1_is_zero'], h['grad_lam1_norm']
    assert h['trace_is_zero'], h['max_abs_trace']
    assert h['cluster_size'] == 2 and h['k_is_lowest_of_cluster']

    Ms = rep['Ms']
    A, C = Ms[:, 0, 0], Ms[:, 1, 1]
    assert np.max(np.abs(A + C)) < 1e-9*np.abs(A).max(), 'traceless means C = -A'


@pytest.mark.slow
def test_first_order_is_automatic_and_says_so(square):
    """The negative result. `Q` is negative semidefinite of rank 2 for structural reasons, so
    the verdict must not claim to certify anything."""
    fam, p, solver, lams, _ = square
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rep = first_order_report(fam, p, solver, lams, k=1)
    w = rep['Q_eigenvalues']
    assert w.max() <= 1e-8*np.abs(w).max(), w
    assert int(np.sum(np.abs(w) > 1e-6*np.abs(w).max())) == 2, f'rank should be 2: {w}'
    assert rep['null_dim'] == fam.n_params - 2
    assert rep['verdict'] == 'first order is automatic'
    assert 'certified' not in rep


@pytest.mark.slow
def test_second_order_is_where_the_content_is(square):
    """Along the directions first order cannot see, the ratio falls -- measured, not certified,
    and the routine's own verdict string says which."""
    fam, p, solver, lams, ratio = square
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rep = first_order_report(fam, p, solver, lams, k=1)
        probe = second_order_probe(ratio, p, rep['null_basis'])
    assert probe['n_dirs'] == fam.n_params - 2
    assert probe['worst_curvature'] < 0, probe['curvatures']
    assert probe['verdict'] == 'consistent with a local maximum'
    assert abs(probe['F0'] - 2.5) < 1e-9, probe['F0']


@pytest.mark.slow
def test_the_triangle_has_no_null_directions(square):
    """N = 3 is the one case where first order IS decisive: `n_p = 2` equals the rank of `Q`, so
    the null space is empty and `Q` is strictly negative definite."""
    fam = VertexPolygon(3, perim=PERIM)
    p = param_vector(_regular(3), PERIM)
    dom = fam.lappy_domain(p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, weyl_est(6, dom), target=1e-13)
        s = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom),
                                       basis=BP.realize(plan, dom), prec=1e-12)
        lams = np.asarray(Eigenproblem(dom, eval_solver=s, precision=1e-12).solve(3), dtype=float)
        rep = first_order_report(fam, p, s, lams, k=1)
    assert abs(rep['ratio'] - 7/3) < 1e-9, rep['ratio']      # equilateral: exactly 7/3
    assert rep['null_dim'] == 0
    assert rep['Q_eigenvalues'].max() < 0


# ── Tracked eigenvalues must stay ORDERED ───────────────────────────────────────

@pytest.mark.slow
def test_a_ratio_run_never_exceeds_the_ppw_bound():
    """A theorem as a regression test, and it caught a real bug.

    Ashbaugh--Benguria: `lam_2/lam_1 <= (j_11/j_01)^2 = 2.5387340` for EVERY bounded connected
    planar domain, with equality only for the disk. So any reported value above it is an error,
    whatever it looks like locally.

    Maximizing the ratio over quadrilaterals used to report 2.6936 on its third iterate.
    `SpectrumEvaluator` tracks each eigenvalue from its own value at the previous iterate, which
    correctly follows a mode THROUGH a crossing -- and then handed back
    `[19.855, 50.807, 48.444]`, unsorted, so an objective reading `lams[1]` got the third mode.
    The true ratio there was 2.2877.
    """
    from douse import SpectrumEvaluator, ObjectiveFunction, RatioObjective
    from douse.objective import RatioObjective as RO
    DISK = (3.8317059702/2.4048255577)**2

    rng = np.random.default_rng(0)
    v = _regular(4)
    v = v + 0.13*(rng.normal(size=4) + 1j*rng.normal(size=4))*np.abs(v).mean()

    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    obj = ObjectiveFunction(ev, RO(1))
    p, alpha = param_vector(v, PERIM), 0.01

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val, g = obj(p)
        for _ in range(10):
            for _ in range(40):
                q = p - alpha*g
                if fam.is_valid(q):
                    v2, g2 = obj(q)
                    if np.isfinite(v2) and v2 < val:
                        break
                alpha *= 0.5
            else:
                break
            p, val, g = q, v2, g2
            alpha *= 1.6
            lams = ev._last[1]
            assert np.all(np.diff(lams) >= -1e-12), f'tracked eigenvalues out of order: {lams}'
            assert -val <= DISK + 1e-9, (
                f'reported lam_2/lam_1 = {-val:.6f}, above the Payne--Polya--Weinberger bound '
                f'{DISK:.6f} that every planar domain obeys')


@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_verify_multiplicity_rejects_a_collapse():
    """Sorting cannot see two seeds landing on one mode -- the array still looks like a
    legitimate degenerate spectrum. The tension spectrum is what tells them apart."""
    from douse import LappySpectrumEvaluator as SpectrumEvaluator

    class _FakeSolver:
        def __init__(self, tensions):
            self._t = np.asarray(tensions)

        def tensions(self, lam):
            return self._t

    ev = SpectrumEvaluator(VertexPolygon(4, perim=PERIM), n_eigs=3, track_ttol=1e-8)
    lams = np.array([10.0, 20.0, 20.0])
    # a genuine double: the first TWO tensions are small
    assert ev._verify_multiplicity(_FakeSolver([1e-12, 1e-12, 0.5]), lams)
    # a collapse: only the first is small, so the pair is one mode counted twice
    assert not ev._verify_multiplicity(_FakeSolver([1e-12, 0.4, 0.7]), lams)
    # no coincident group at all: nothing to check
    assert ev._verify_multiplicity(_FakeSolver([0.9, 0.9, 0.9]), np.array([1.0, 2.0, 3.0]))
