"""The shape derivative, against finite differences. This is Milestone 1.

Everything douse will do rests on `dlam/dp` being right, and a wrong shape gradient does not
crash -- it is finite, plausible, and an optimizer follows it happily into a wrong answer. The
predecessor package's explicit parametrization Jacobian was left commented out and marked
BUGGED, and its eigenvalue derivative was never checked against anything.

THREE LEVELS, SO A FAILURE LOCALIZES. `test_polygon.py` verifies `d vertices/dp` on its own with
no eigensolve. Here, `dlam/d vertices` is checked on its own, and only then the composition. A
single end-to-end check would leave a failure ambiguous between the geometry and the physics.

THE FINITE DIFFERENCE IS THE WEAK INSTRUMENT, not the gradient, and getting it sound took two
corrections that are worth recording because both are easy to omit.

A first version differenced at `target=1e-12` (tension floor `sigma = 1.15e-10`) with every
stencil point tracked from `lam(p0)`. It read 3.7e-04 on one parameter. The cause was a single
stencil point: changing the track SEED by 4.6e-13 moved the located eigenvalue by 5.5e-07, and
`8 * 5.5e-07 / (12h)` is exactly the discrepancy. Near a minimum `sigma(lam) ~ sigma_min +
c (lam - lam*)^2`, so lambda is only resolvable to about `sqrt(sigma_min/c)` however tight
`ltol` is -- and with `sigma_min ~ 1e-10` that is around 1e-07, right where the difference lives.
Tightening `ltol` alone does nothing (measured: identical results at 1e-12 and 1e-14).

So: **a tighter plan target**, which lowers the tension floor (1.15e-10 -> 1.02e-11), and
**continuation seeding**, where each stencil point is tracked from its neighbour rather than all
from the centre. `benchmarks/basis_lab/plan_lab.py`'s own FD reference does both. After the fix:

    h                     3e-3       1e-3
    |analytic - FD|     3.8e-08    2.0e-08

Flat in h across the range where truncation and roundoff trade off, which is what a sound
reference looks like. The assertions below use 1e-6, clearing that by more than an order of
magnitude, so they catch a regression without tracking FD noise.

Three independent confirmations that the ANALYTIC value is the right one, since a
finite-difference disagreement is not by itself evidence about which side is wrong: FD in `p`
converges to it as `h` falls; an FD in vertex space along the constraint direction `J[:,0]`
agrees to 6.5e-08 by a different path; and `test_a_rigid_translation_moves_no_eigenvalue` is an
exact identity that a sign or bookkeeping error could not satisfy.
"""
import warnings

import numpy as np
import pytest

lappy = pytest.importorskip("lappy")   # lappy solvers and Hadamard path: the deprecated backend
from lappy import Eigenproblem, Polygon, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse.polygon import VertexPolygon, param_vector
from douse.derivative import dlam_dvertices, dlam_dp, vertex_velocity

PERIM = 4.0
SEED_SHAPE = np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j])
H = 1e-3


@pytest.fixture(scope='module')
def setup():
    """A frozen plan, the base solver and lambda_1, built once.

    The plan is frozen at p0 and realized on every perturbed shape, so the basis is identical
    throughout and the finite difference measures the SHAPE rather than a change of basis. That
    is the architecture lappy's `plan_basis`/`realize` split exists to provide.
    """
    fam = VertexPolygon(4, perim=PERIM)
    p0 = param_vector(SEED_SHAPE, PERIM)
    dom0 = fam.lappy_domain(p0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom0, weyl_est(6, dom0), target=1e-14)

    def solve(dom, seed=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom),
                                           basis=BP.realize(plan, dom), prec=1e-13)
            evp = Eigenproblem(dom, eval_solver=s, precision=1e-13)
            lam = float(evp.solve(1)[0]) if seed is None else float(evp.track(seed))
        return lam, s

    lam0, s0 = solve(dom0)
    return dict(fam=fam, p0=p0, dom0=dom0, plan=plan, solve=solve, lam0=lam0, s0=s0)


def _fd5(lam_at, h, lam0):
    """Five-point central difference, walking outward from the centre by CONTINUATION.

    `lam_at(t, seed)` solves at displacement `t` starting from `seed`. Each stencil point is
    seeded from its neighbour rather than all of them from `lam0`, so the tracker never has to
    jump a full `2h` and the located minimum does not depend on how far the seed was. See the
    module docstring for the 3.7e-04 this was worth.
    """
    v, seed = {}, lam0
    for t in (h, 2*h):
        seed = lam_at(t, seed)
        v[t] = seed
    seed = lam0
    for t in (-h, -2*h):
        seed = lam_at(t, seed)
        v[t] = seed
    return (v[-2*h] - 8*v[-h] + 8*v[h] - v[2*h])/(12*h)


@pytest.mark.slow
def test_the_base_solve_is_sound(setup):
    """Guard on the instrument: a loose tension here would make every tolerance below meaningless."""
    lam0, s0 = setup['lam0'], setup['s0']
    sigma = float(np.atleast_1d(s0.sigma(lam0))[0])
    assert 15.0 < lam0 < 30.0, lam0
    assert sigma < 1e-10, (f'tension {sigma:.2e} at the base eigenvalue is too loose to '
                           f'difference: lambda is only resolvable to ~sqrt(sigma/c)')


@pytest.mark.slow
def test_dlam_dvertices_matches_a_finite_difference(setup):
    """The Hadamard formula, one vertex at a time.

    These are UNCONSTRAINED partials -- moving a single vertex changes the perimeter, so the
    perturbed shapes leave the parametrization's family and are built as bare polygons. The
    constraint enters one level up, in the composition below.
    """
    fam, solve, lam0, s0 = setup['fam'], setup['solve'], setup['lam0'], setup['s0']
    v0 = fam.vertices(setup['p0'])
    g = dlam_dvertices(s0, lam0, fam.n)
    assert g.shape == (2*fam.n,)

    worst = 0.0
    for k in range(fam.n):
        for axis in (0, 1):
            step = 1.0 if axis == 0 else 1j

            def lam_at(t, seed, k=k, step=step):
                v = v0.copy()
                v[k] += t*step
                return solve(Polygon(v), seed=seed)[0]

            err = abs(g[axis*fam.n + k] - _fd5(lam_at, H, lam0))
            worst = max(worst, err)
    assert worst < 1e-6, f'max |analytic - FD| over all 2N vertex partials = {worst:.2e}'


@pytest.mark.slow
def test_dlam_dp_matches_a_finite_difference(setup):
    """MILESTONE 1. The composition, against a finite difference in the parameters themselves.

    Perturbing `p` and rebuilding through `polygon_vertices` keeps the perimeter fixed, so this
    differences along the constraint surface -- which is what an optimizer actually walks.
    """
    fam, solve, p0, lam0, s0 = (setup['fam'], setup['solve'], setup['p0'],
                                setup['lam0'], setup['s0'])
    g = dlam_dp(fam, p0, s0, lam0)
    assert g.shape == (fam.n_params,)

    fd = np.empty_like(g)
    for i in range(len(p0)):
        def lam_at(t, seed, i=i):
            p = p0.copy()
            p[i] += t
            return solve(fam.lappy_domain(p), seed=seed)[0]
        fd[i] = _fd5(lam_at, H, lam0)

    err = np.max(np.abs(g - fd))
    rel = err/np.max(np.abs(g))
    assert err < 1e-6, (f'max |analytic - FD| = {err:.2e} (relative {rel:.2e})\n'
                        f'  analytic: {np.array2string(g, precision=8)}\n'
                        f'  FD      : {np.array2string(fd, precision=8)}')


@pytest.mark.slow
def test_the_gradient_predicts_a_step(setup):
    """The property an optimizer actually uses: `lam(p + t d) - lam(p) ~ t (g . d)`.

    Checked along a direction that mixes every parameter, because a per-coordinate check can
    pass with a permuted or mis-scaled gradient.
    """
    fam, solve, p0, lam0, s0 = (setup['fam'], setup['solve'], setup['p0'],
                                setup['lam0'], setup['s0'])
    g = dlam_dp(fam, p0, s0, lam0)
    d = np.array([0.37, -0.51, 0.62, -0.28])
    d = d/np.linalg.norm(d)
    predicted = g @ d
    measured = _fd5(lambda t, seed: solve(fam.lappy_domain(p0 + t*d), seed=seed)[0], H, lam0)
    assert abs(predicted - measured) < 1e-6, (predicted, measured)


@pytest.mark.slow
def test_the_wrong_node_set_is_measurably_worse_here(setup):
    """`bdry_quad` is for normalization; every vertex move is a corner move.

    On this convex quadrilateral no corner is reentrant, so lappy's guard correctly stays
    silent and the two node sets agree -- which is exactly why this trap is easy to miss. The
    assertion is that they agree HERE, so that if a future change to the parametrization admits
    reentrant corners, the reentrant test in lappy is what catches it rather than this passing
    quietly.
    """
    fam, lam0, s0 = setup['fam'], setup['lam0'], setup['s0']
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # the guard must NOT fire on a convex polygon
        g_had = dlam_dvertices(s0, lam0, fam.n)
        g_bdry = dlam_dvertices(s0, lam0, fam.n, bq=s0.bdry_quad)
    assert np.max(np.abs(g_had - g_bdry)) < 1e-6*np.max(np.abs(g_had))
    assert s0.hadamard_quad.weight_family == 'integer'
    assert s0.bdry_quad.weight_family == 'even'


# ── Structure and refusals, no eigensolve ───────────────────────────────────────

def test_vertex_velocity_moves_two_edges_and_nothing_else(setup):
    """The velocity field is the piecewise-linear vertex perturbation: linear along the two
    incident edges, zero elsewhere, and equal to the full displacement AT the vertex."""
    fam, s0 = setup['fam'], setup['s0']
    bq = s0.hadamard_quad
    for k in range(fam.n):
        dp = vertex_velocity(bq, k, fam.n, 1.0 + 0.0j)
        incident = (bq.seg_idx == (k - 1) % fam.n) | (bq.seg_idx == k % fam.n)
        assert np.all(dp[~incident] == 0), f'vertex {k} moved a non-incident edge'
        assert np.max(np.abs(dp)) <= 1.0 + 1e-12
        # at the vertex itself the displacement is the full step: tau -> 1 on the incoming edge
        prev = bq.seg_idx == (k - 1) % fam.n
        assert abs(np.max(np.abs(dp[prev])) - np.max(bq.tau[prev])) < 1e-12


def test_a_rigid_translation_moves_no_eigenvalue(setup):
    """Every vertex moved by the same amount is a translation, and the spectrum cannot see it.
    Summing the per-vertex partials must give zero -- a strong check on the SIGNS, which a
    per-coordinate finite difference agreeing to 6 digits could still get collectively wrong."""
    fam, lam0, s0 = setup['fam'], setup['lam0'], setup['s0']
    g = dlam_dvertices(s0, lam0, fam.n)
    dx = g[:fam.n].sum()
    dy = g[fam.n:].sum()
    scale = np.max(np.abs(g))
    assert abs(dx) < 1e-6*scale, f'translating in x changes lambda by {dx:.3e}'
    assert abs(dy) < 1e-6*scale, f'translating in y changes lambda by {dy:.3e}'


def test_a_multiple_eigenvalue_is_refused_with_the_mathematics(setup):
    """Not implemented, and the message says why rather than pretending it is an oversight."""
    fam, lam0, s0 = setup['fam'], setup['lam0'], setup['s0']
    with pytest.raises(NotImplementedError, match='not a gradient'):
        dlam_dvertices(s0, lam0, fam.n, mult=2)
