"""Maximizing lam_k/lam_1 through a degeneracy.

The acceptance test is the square. Backtracking ascent stalls 5.70e-04 short of it with a
gradient that never falls -- the signature of grinding against a corner -- because at the square
lam_2 = lam_3 exactly and no gradient exists to vanish. The trust-region cluster method converges
to 6.9e-09.

THREE BUGS THIS FILE EXISTS TO PREVENT, all of them found by running the thing rather than by
reading it, and all of them producing plausible wrong numbers rather than crashes:

1. The model ignored the actual splitting within the eps-active cluster, so the trust radius
   collapsed while still 1.9e-03 short.
2. `Lambda` was then added to `M` in the WRONG BASIS -- `eigenfunction_coef(mult=2)` returns an
   arbitrary Loewdin basis, which does not diagonalize the unperturbed splitting. The model's
   predicted direction was wrong, showing up as trust-region ratios that stayed negative
   (-0.24 to -1.23) as the radius shrank where a correct first-order model gives +1.
3. Optimality was tested on the raw model gain, which scales with the trust radius, so the
   method declared "first-order optimal" whenever the radius had shrunk -- at a point with an
   O(1) directional derivative.

...AND A FOURTH, found the same way once the subproblem got a reference to be measured against.
The concave branch used to introduce `t` and impose `tr >= 0`, `det >= 0`. At `j = 0` the
determinant constraint is degenerate at the natural start `z = 0`, and SLSQP simply stopped
there: **45 per cent short of the true optimum at Delta = 0.05**, and worse as the radius shrank.
Nothing detected it, because a trust-region method fed a too-small step still makes progress --
it just makes less of it, which reads as slow convergence rather than as a bug.

Hence the tests below that compare each branch against an INDEPENDENT reference: brute-force
multi-start maximization of the exact model for the nonconvex branches, and the `cvxpy`
semidefinite program for the concave one. They need no eigensolver, so they run in about a
second and can be trusted to fail for one reason only.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from douse import VertexPolygon, param_vector, SpectrumEvaluator
from douse.cluster_opt import (CASE_CERTIFIED, local_model, trust_region_step, maximize_ratio,
                               model_gain, _active_cluster, _case_record, _case_row, _step_concave_sdp,
                               _subproblem)

PERIM = 4.0
PPW = (3.8317059702/2.4048255577)**2


def _regular(N, perim=PERIM):
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


def _perturbed(N, jitter, seed=0):
    rng = np.random.default_rng(seed)
    v = _regular(N)
    return v + jitter*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()


@pytest.mark.slow
def test_the_square_is_reached_through_the_degeneracy():
    """The acceptance test. Descent stalls at 5.70e-04; this must do far better, and must end AT
    the degeneracy rather than short of it."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(_perturbed(4, 0.06), PERIM), k=1, max_iter=120)

    assert abs(res.fun - 2.5) < 1e-6, (res.fun, res.message)
    assert res.fun <= PPW + 1e-9, f'{res.fun} exceeds the PPW bound {PPW}'
    assert res.gap < 1e-5, f'should finish AT the degeneracy, gap = {res.gap:.2e}'
    assert res.n_cluster_steps > 0, 'the cluster model was never used'
    c = res.final_case
    assert c['row'] == 'lowest_2' and c['certified'] and c['at_endpoint'], c
    assert c['cluster'] == (1, 3) and c['resolved_m'] == 2 and c['resolved_j'] == 0, c

    v = fam.vertices(res.x)
    ang = np.degrees([np.pi - np.angle((v[(i+1) % 4]-v[i])/(v[i]-v[i-1])) for i in range(4)])
    assert np.ptp(ang) < 0.5, f'angles {ang} are not square'


@pytest.mark.slow
def test_the_ratio_never_exceeds_the_ppw_bound_during_a_run():
    """A theorem as an invariant, checked at EVERY iterate rather than only at the end.

    This is how the stale-objective bug surfaced: the optimizer reported 2.5400 for a pentagon
    whose true ratio was 2.5124, because `F` was not recomputed when `lams` was refreshed and the
    refresh continued from a rejected trial point.
    """
    fam = VertexPolygon(5, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    seen = []

    def cb(p, F, model, delta, rho):
        seen.append(F)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(_perturbed(5, 0.06), PERIM), k=1,
                             max_iter=40, callback=cb)
    assert seen, 'no iterations ran'
    worst = max(seen + [res.fun])
    assert worst <= PPW + 1e-9, f'reported {worst:.9f}, above the PPW bound {PPW:.9f}'


@pytest.mark.slow
def test_the_objective_and_the_eigenvalues_stay_consistent():
    """`res.fun` must be the ratio OF `res.lams`. They drifted apart once, and the eigenvalues
    were the half that stayed right -- so a caller trusting `fun` got a number no shape had."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(_perturbed(4, 0.06), PERIM), k=1, max_iter=30)
    assert abs(res.fun - res.lams[1]/res.lams[0]) < 1e-12, (res.fun, res.lams)


@pytest.mark.slow
def test_the_model_switches_to_the_cluster_as_the_gap_closes():
    """`eps_rel` is an eps-ACTIVE set: smooth while the modes are apart, cluster once they are
    within tolerance. Both branches must be exercised by a real run."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    modes = []

    def cb(p, F, model, delta, rho):
        modes.append(model['mode'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        maximize_ratio(ev, param_vector(_perturbed(4, 0.10), PERIM), k=1, max_iter=60,
                       callback=cb)
    assert 'smooth' in modes and 'cluster' in modes, set(modes)
    assert modes.index('cluster') > modes.index('smooth'), 'cluster should come second'


@pytest.mark.slow
def test_the_convex_case_is_answered_and_the_square_is_not_optimal_for_it():
    """`lam_3` is the HIGHEST member of its cluster at the square, which used to be refused.

    Exchanging the two maxima in `lam_max(Lambda+M(d)) = max_u [...]` turns the subproblem into a
    search over the unit circle, so it is answered exactly now. And the answer is a finding: at
    the exact square the model gain for `lam_3/lam_1` is not zero but O(1e-02), so the square is
    NOT a first-order local maximizer of `lam_3/lam_1`, even though it is one for `lam_2/lam_1`
    and the two ratios have the same value there (5/2).
    """
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, warn_gap=0.0, degenerate_tol=0.0)
    p = param_vector(_regular(4), PERIM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, solver = ev.eigenvalues(p)
        m2 = local_model(fam, p, solver, lams, k=1, eps_rel=1e-6)
        m3 = local_model(fam, p, solver, lams, k=2, eps_rel=1e-6)
        _, gain2 = trust_region_step(m2, 0.02)
        _, gain3 = trust_region_step(m3, 0.02)

    assert (m2['m'], m2['j']) == (2, 0), 'lam_2 is the LOWEST of its double'
    assert (m3['m'], m3['j']) == (2, 1), 'lam_3 is the HIGHEST of the same double'
    assert gain2 < 1e-9, f'the square IS first-order optimal for lam_2/lam_1, gain {gain2:.2e}'
    assert gain3 > 1e-3, f'the square is NOT, for lam_3/lam_1; gain came back {gain3:.2e}'


def test_an_evaluator_that_refuses_degeneracies_is_rejected():
    """The guard that normally protects a smooth method would defeat this one."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3)          # default degenerate_tol > 0
    with pytest.raises(ValueError, match='degenerate_tol=0'):
        maximize_ratio(ev, param_vector(_regular(4), PERIM), k=1)


def test_the_trust_region_step_scales_with_the_radius():
    """A smooth-mode sanity check with no solver: steepest ascent to the boundary, and a gain
    proportional to the radius -- which is why criticality has to be measured as gain/radius."""
    model = dict(mode='smooth', grad=np.array([3.0, 4.0]))
    for delta in (0.1, 1.0):
        d, gain = trust_region_step(model, delta)
        assert abs(np.linalg.norm(d) - delta) < 1e-12
        assert abs(gain - 5.0*delta) < 1e-12


# ---------------------------------------------------------------------------------------------
# The subproblem, against independent references. No eigensolver: these build synthetic cluster
# models directly, so a failure is a failure of the subproblem solver and of nothing else.

def _synthetic(m, n_p, j, seed):
    """A cluster model with `lam_k` at position `j` of an m-fold cluster.

    `Lam`'s entries are offsets relative to `lam_k`, so entry `j` is zero by construction --
    which is exactly the degeneracy that made the old determinant-constrained formulation stall.
    """
    rng = np.random.default_rng(seed)
    Ms = rng.normal(size=(n_p, m, m))
    Ms = Ms + np.transpose(Ms, (0, 2, 1))
    off = np.sort(rng.uniform(0, 1e-2, size=m))
    return dict(mode='cluster', m=m, j=j, Lam=np.diag(off - off[j]), Ms=Ms,
                g1=rng.normal(size=n_p), lam_1=19.7, lam_k=49.3, cluster=(0, m))


def _brute(model, delta, n_starts=40, seed=1):
    """Multi-start maximization of the EXACT model over the trust region."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    n_p = len(model['g1'])
    cons = [dict(type='ineq', fun=lambda d: delta**2 - d @ d)]
    best = -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(n_starts):
            d0 = rng.normal(size=n_p)
            d0 *= delta/np.linalg.norm(d0)
            r = minimize(lambda d: -model_gain(model, d), d0, method='SLSQP',
                         constraints=cons, options=dict(maxiter=200, ftol=1e-14))
            d = r.x*min(1.0, delta/np.linalg.norm(r.x))
            best = max(best, model_gain(model, d))
    return best


@pytest.mark.parametrize('m', [2, 3])
@pytest.mark.parametrize('delta', [0.02, 0.3])
def test_the_convex_reduction_is_exact(m, delta):
    """`lam_k` highest of its cluster: the sphere reduction should be EXACT, not approximate.

    `max_d [lam_max(Lam + M(d))*lam_1 - lam_k g1.d]` equals `max_{||u||=1}` of a norm, by
    exchanging the two maxima -- so it is a search over `S^{m-1}` and there is nothing to
    approximate. Anything worse than brute force here means the exchange was implemented wrong.
    """
    model = _synthetic(m, 5, m - 1, seed=17*m)
    d, gain = trust_region_step(model, delta)
    assert np.linalg.norm(d) <= delta*(1 + 1e-9)
    assert gain >= _brute(model, delta) - 1e-10


@pytest.mark.parametrize('delta', [0.01, 0.05, 0.3])
def test_the_concave_step_matches_the_semidefinite_program(delta):
    """The 2x2 fast path must agree with the SDP it is a shortcut for.

    This is the test that would have caught the 45-per-cent shortfall: the LMI formulation and
    the linear-minus-a-norm one describe the same program, so they must return the same value,
    and `cvxpy` solves the LMI without SLSQP's start-point problem.
    """
    cvxpy = pytest.importorskip('cvxpy')                                       # noqa: F841
    model = _synthetic(2, 5, 0, seed=23)
    d, gain = trust_region_step(model, delta)
    ref = model_gain(model, _step_concave_sdp(model, delta))
    assert np.linalg.norm(d) <= delta*(1 + 1e-9)
    assert gain >= ref - 1e-4*abs(ref), f'{gain:.9f} against the SDP\'s {ref:.9f}'


@pytest.mark.parametrize('m,j', [(3, 1), (4, 1), (4, 2)])
def test_the_interior_step_is_within_a_whisker_of_brute_force(m, j):
    """An interior `sigma_j` is neither convex nor concave, so the exact model is maximized
    directly. Bracketing it between the two extremes was tried first and measured 10 to 43 per
    cent short; this must not be."""
    pytest.importorskip('cvxpy')
    model = _synthetic(m, 5, j, seed=31*m + j)
    _, gain = trust_region_step(model, 0.1)
    ref = _brute(model, 0.1)
    assert gain >= ref - 1e-4*abs(ref), f'{gain:.9f} against brute force\'s {ref:.9f}'


@pytest.mark.parametrize('m,j', [(2, 0), (2, 1), (3, 0), (3, 2)])
def test_the_reported_gain_is_the_exact_model_at_the_step_taken(m, j):
    """`rho` divides a real improvement by this number, so it has to be the model's value AT the
    returned `d` -- not a surrogate's optimal value, which for the interior branch is a different
    quantity and for a stalled solver is a larger one."""
    model = _synthetic(m, 5, j, seed=7)
    for delta in (0.01, 0.2):
        d, gain = trust_region_step(model, delta)
        assert abs(gain - model_gain(model, d)) < 1e-12
        assert gain >= 0.0


def test_the_active_cluster_is_sized_by_the_trust_radius():
    """`eps_abs` is what ties the eps-active set to the step that could merge its members. A
    radius small enough to merge nothing must leave the cluster a singleton, and the model then
    smooth; a large one must pick up the neighbour."""
    lams = np.array([19.7, 49.30, 49.34])
    assert _active_cluster(lams, 1, 1e-6) == (1, 2)          # nothing merges
    assert _active_cluster(lams, 1, 1e-1) == (1, 3)          # the neighbour is in reach


# ---------------------------------------------------------------------------------------------
# General k, against arithmetic.

# The square (side 1) and the equilateral triangle (side 4/3) both have perimeter 4 and both have
# closed-form Dirichlet spectra: pi^2*(m^2+n^2) and pi^2*(m^2+mn+n^2) with m,n >= 1. So `lam_k`,
# the size of each cluster, and the POSITION of `lam_k` within it are all known exactly, for
# every k -- which is what the general-k machinery needs and what no optimization run can supply,
# since the answers those produce are the research question.
_EXACT = {
    'square': (np.array([0, 1, 1+1j, 1j], dtype=complex),
               np.pi**2*np.array([2., 5, 5, 8, 10, 10]),
               [(2, 0), (2, 1), (1, 0), (2, 0), (2, 1)]),
    'eq_tri': (PERIM/3*np.array([0, 1, 0.5+0.5j*np.sqrt(3)], dtype=complex),
               np.pi**2*np.array([3., 7, 7, 12, 13, 13]),
               [(2, 0), (2, 1), (1, 0), (2, 0), (2, 1)]),
}


@pytest.mark.slow
@pytest.mark.parametrize('name', sorted(_EXACT))
def test_every_k_gets_the_right_cluster_and_the_exact_ratio(name):
    """For k = 2..6 the spectrum, the cluster size m, and the position j must all be exact.

    `j` is the whole content: it decides which branch of `trust_region_step` runs, and getting it
    wrong silently sends a convex case to the concave solver, which returns a plausible number.
    Here it is checked against arithmetic rather than against another run of the same code.
    """
    verts, exact, structure = _EXACT[name]
    fam = VertexPolygon(len(verts), perim=PERIM)
    p = param_vector(verts, PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=len(exact), warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=len(exact) + 4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, solver = ev.eigenvalues(p)
        models = [local_model(fam, p, solver, lams, k=k, eps_rel=1e-6)
                  for k in range(1, len(exact))]

    assert np.allclose(lams, exact, rtol=1e-11), np.c_[lams, exact]
    for k, (model, (m, j)) in enumerate(zip(models, structure), start=1):
        got = (model['m'], model['j'])
        assert got == (m, j), f'lam_{k+1}: cluster (m, j) is {got}, should be {(m, j)}'
        assert abs(model['ratio'] - exact[k]/exact[0]) < 1e-10

    # every branch must produce a step rather than refusing -- that is the point of the rewrite
    for model in models:
        d, gain = trust_region_step(model, 0.02)
        assert np.all(np.isfinite(d)) and gain >= 0.0


@pytest.mark.slow
def test_the_returned_ratio_is_cold_solved_not_tracked():
    """A run that converges ONTO a degeneracy cannot trust its own tracked endpoint.

    Measured before `verify_final` existed: the N=4 run reported 2.500000005354, which is
    5.4e-09 ABOVE 5/2 and therefore impossible if the square maximizes `lam_2/lam_1`. The shape
    was fine -- a square to 2.1e-06 in the sides -- and a cold solve at that very `p` gave
    2.499999994645, on the correct side. Only the continuation was wrong, and only at the last
    place a run looks.

    So `fun` must be BELOW 5/2, and it must differ from the tracked value it replaced.
    """
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(_perturbed(4, 0.06), PERIM), k=1, max_iter=120)

    assert res.verified
    assert res.fun <= 2.5, f'{res.fun:.12f} is above 5/2, which no quadrilateral reaches'
    assert abs(res.fun - 2.5) < 1e-7, res.fun
    assert abs(res.fun - res.lams[1]/res.lams[0]) < 1e-12, 'fun must be the ratio OF lams'


@pytest.mark.slow
def test_the_second_order_model_reaches_the_same_square():
    """`quasi_newton` must change the PATH and not the ANSWER.

    The second-order model is a better local model of the same objective, so the endpoint is a
    property of the problem and must survive turning it on. Run against the same start as
    `test_the_square_is_reached_through_the_degeneracy`, which is the acceptance test the
    first-order method is held to.
    """
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, param_vector(_perturbed(4, 0.06), PERIM), k=1, max_iter=120,
                             quasi_newton=True)

    assert abs(res.fun - 2.5) < 1e-6, (res.fun, res.message)
    assert res.fun <= PPW + 1e-9, f'{res.fun} exceeds the PPW bound {PPW}'
    assert res.gap < 1e-5, f'should finish AT the degeneracy, gap = {res.gap:.2e}'
    assert res.n_qn_steps > 0, 'the second-order step was never taken'


@pytest.mark.slow
def test_the_second_order_model_does_not_cost_iterations():
    """It is allowed to help or to do nothing; it is not allowed to make the run longer.

    Stated as an inequality with slack rather than as a speed-up, because the size of the gain is
    a property of the landscape -- measured at 106 -> 79 iterations on `max lam_3/lam_1` over
    quadrilaterals, and much smaller on runs that spend all their time in cluster mode, where the
    only second-order content is the quotient's own curvature.
    """
    fam = VertexPolygon(4, perim=PERIM)
    out = {}
    for qn in (False, True):
        ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[qn] = maximize_ratio(ev, param_vector(_perturbed(4, 0.06), PERIM), k=1,
                                     max_iter=200, quasi_newton=qn)
    assert abs(out[True].fun - out[False].fun) < 1e-6, (out[True].fun, out[False].fun)
    assert out[True].nit <= out[False].nit + 5, (out[True].nit, out[False].nit)


def test_the_stopping_certificate_stays_first_order():
    """The stop is measured against `model_gain`, never against `model_gain2`.

    If it were measured against the quadratic model, a run near a maximum would stop early: the
    Newton step goes interior there, so its predicted gain falls like `|grad|^2` while `delta`
    stays at `delta_max`, and `gain/delta` would clear `gain_tol` at `|grad| ~ 3e-04`. This pins
    the property that makes that impossible -- the two models agree to first order, so the
    criticality measure is the same number either way as the radius shrinks.
    """
    rng = np.random.default_rng(0)
    n_p = 6
    g1, gk = rng.normal(size=n_p), rng.normal(size=n_p)
    lam_1, lam_k = 20.3, 64.1
    model = dict(mode='smooth', lam_1=lam_1, lam_k=lam_k, g1=g1, gk=gk, m=1, j=0,
                 cluster=(1, 2), grad=(gk*lam_1 - lam_k*g1)/lam_1**2)
    grad_norm = float(np.linalg.norm(model['grad']))
    for delta in (1e-2, 1e-4, 1e-6):
        d, gain = trust_region_step(model, delta)
        assert gain/delta == pytest.approx(grad_norm, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# The resolution floor on the eps-active set, and the criticality plateau.

def test_the_active_set_never_shrinks_below_the_resolution_floor():
    """The bug this fixes ended 8 of the first 10 multi-start runs, including every run that found
    the BEST answer.

    `eps_abs = Delta*scale` is right on the way in and wrong at the end. At the N=4
    `max lam_4/lam_1` optimum the top pair is split by 1.14e-10 RELATIVE -- numerically identical
    against a basis resolving to ~1e-10 -- and `scale` is 123, so below `Delta = 9.4e-11` the
    cluster stopped being seen. The model then reverted to `m = 1` and returned the
    simple-eigenvalue gradient at a degenerate point, which is a DIFFERENT QUANTITY (see
    `DegenerateSpectrum`), reporting criticality 1.35 where the truth is 1.1e-06. Every step then
    failed and the radius ground to 5.68e-15.
    """
    from douse.cluster_opt import _eps_width

    lam_k = 101.2178122523
    # what the trust radius asks for, at the radii a converged run actually reaches
    assert _eps_width(lam_k, 1e-3, 1e-10*123, 0.0) == pytest.approx(1.23e-08)
    # ... and with the floor, it cannot go below it
    assert _eps_width(lam_k, 1e-3, 1e-12*123, 1e-9) == pytest.approx(1e-9*lam_k)
    assert _eps_width(lam_k, 1e-3, 0.0, 1e-9) == pytest.approx(1e-9*lam_k)
    # the floor must not INFLATE a legitimately wide request
    assert _eps_width(lam_k, 1e-3, 5.0, 1e-9) == pytest.approx(5.0)


def test_the_floor_keeps_a_numerically_degenerate_pair_clustered_at_any_radius():
    """The property that matters, stated on the actual endpoint spectrum."""
    from douse.cluster_opt import _active_cluster, _eps_width

    lams = np.array([23.6302578136, 55.3495724946, 57.3573811503,
                     101.2178122523, 101.2178122638])
    k, scale = 3, 123.0
    for delta in (1e-2, 1e-6, 1e-10, 1e-12, 5.68e-15):
        lo, hi = _active_cluster(lams, k, _eps_width(lams[k], 1e-3, delta*scale, 1e-9))
        assert hi - lo == 2, f'cluster lost at delta={delta:.1e}'
    # with no floor it IS lost, which is the bug
    lo, hi = _active_cluster(lams, k, _eps_width(lams[k], 1e-3, 1e-12*scale, 0.0))
    assert hi - lo == 1


def test_the_floor_admits_exactly_the_neighbours_closer_than_itself():
    """The floor is a resolution statement, so it must pull in the neighbours that are
    numerically coincident and no others -- a safety net, not a widening.

    The asymmetry that justifies having one at all is not testable here: it is that `Lam` carries
    a real splitting exactly, so a too-wide active set costs work and not accuracy, while a
    too-narrow one substitutes a different quantity for the gradient (see `DegenerateSpectrum`).
    What is testable is that the floor does only what it says.
    """
    from douse.cluster_opt import _active_cluster, _eps_width

    lams = np.array([20.0, 60.0, 60.0 + 1e-7, 60.0 + 3e-3])
    # 1e-8 relative is 6e-07 absolute: reaches the 1e-07 neighbour, not the 3e-03 one
    assert _active_cluster(lams, 1, _eps_width(lams[1], 0.0, 0.0, 1e-8)) == (1, 3)
    # 1e-9 relative is 6e-08: reaches neither
    assert _active_cluster(lams, 1, _eps_width(lams[1], 0.0, 0.0, 1e-9)) == (1, 2)
    # and an ordinary eps_rel request still reaches everything within 1e-3 of lam_k
    assert _active_cluster(lams, 1, _eps_width(lams[1], 1e-3, None, 1e-9)) == (1, 4)


def test_chladni_clusters_are_a_floor_under_the_active_set():
    """chladni links levels whose eigenfunctions are entangled into a `Cluster` (2026-11-05);
    their members' own derivatives are not determined, so the eps-active set takes a cluster
    whole however narrow `eps_abs` is. Without one, the width alone decides, as before."""
    from chladni.eigensolver import Cluster, Level, Spectrum
    from douse.cluster_opt import _active_cluster
    lam = [10.0, 20.0, 20.0 + 2e-9, 30.0]
    plain = Spectrum(tuple(Level(l, 1, 1e-12) for l in lam))
    assert _active_cluster(lam, 1, 1e-12, spec=plain) == (1, 2)
    levels = tuple(Level(l, 1, 1e-12, cluster=(0 if i in (1, 2) else None))
                   for i, l in enumerate(lam))
    spec = Spectrum(levels, clusters=(Cluster(levels=(1, 2), mult=2, spread=1e-10,
                                              overlap=1e-6),))
    assert _active_cluster(lam, 1, 1e-12, spec=spec) == (1, 3)
    assert _active_cluster(lam, 2, 1e-12, spec=spec) == (1, 3)
    assert _active_cluster(lam, 3, 1e-12, spec=spec) == (3, 4)


def test_hadamard_shortfall_warnings_are_counted_apart():
    """chladni's calibrated Hadamard estimate (#23) warns more often than the faults `n_warnings`
    was meant to flag, so the run counts it separately as well."""
    import warnings
    from douse.cluster_opt import _watch_warnings
    tally = dict(n=0, n_hadamard_short=0, seen=set())
    with _watch_warnings(tally):
        for _ in range(3):
            warnings.warn("the Hadamard matrix at lam = 49.3 is good to 1.1e-10 relative")
        warnings.warn("a level at lam = 12 is unmet")
    assert tally['n'] == 4 and tally['n_hadamard_short'] == 3


# THE CASE A RUN ENDED IN. The rows are read from the same function `_subproblem` dispatches on,
# so these check the labels, the fallback and the record's arithmetic.

@pytest.mark.parametrize('m,j,row', [(2, 0, 'lowest_2'), (3, 0, 'lowest_m'), (4, 0, 'lowest_m'),
                                     (2, 1, 'highest_2'), (3, 2, 'highest_m'), (3, 1, 'middle'),
                                     (5, 2, 'middle')])
def test_each_position_names_its_row(m, j, row):
    assert _case_row(_synthetic(m, 6, j, seed=0), 0.01) == row


def test_only_the_rows_with_a_reduction_are_certified():
    assert {r for r, ok in CASE_CERTIFIED.items() if not ok} == {'highest_m', 'middle',
                                                                 'constrained'}


def test_a_constraint_in_reach_overrides_the_position():
    model = _synthetic(2, 3, 0, seed=0)
    near, far = (np.array([1e-3]), np.eye(3)[:1]), (np.array([1.0]), np.eye(3)[:1])
    assert _case_row(model, 0.01, near) == 'constrained'
    assert _case_row(model, 0.01, far) == 'lowest_2'
    assert _case_row(dict(mode='smooth'), 0.01, near) == 'constrained'
    assert _case_row(dict(mode='smooth'), 0.01) == 'simple'


def test_a_fallback_is_recorded_and_costs_the_certificate(monkeypatch):
    import douse.cluster_opt as co

    def state(model):
        return SimpleNamespace(model=model, lams=np.array([10.0, 20.0, 20.5, 20.75, 30.0]),
                               delta=0.01, lin=None, sub_ok=True)
    model = _synthetic(3, 6, 0, seed=0)
    model.update(cluster=(1, 4), resolved=(1, 4))
    _subproblem(model, 0.01, rng=np.random.default_rng(0))
    assert model['fallback'] is False and _case_record(state(model))['certified']

    monkeypatch.setattr(co, '_step_concave_sdp', lambda model, delta: None)
    _, gain, ok = _subproblem(model, 0.01, rng=np.random.default_rng(0))
    assert ok and gain > 0 and model['fallback'] is True
    c = _case_record(state(model))
    assert c['row'] == 'lowest_m' and c['fallback'] and not c['certified'] and c['solved']


def test_the_record_separates_what_the_radius_saw_from_the_shape():
    """A wide radius takes three eigenvalues; at the floor width only the pair remains, and `k`
    is the upper of the two, so the position differs as well as the size."""
    model = dict(mode='cluster', m=3, j=1, cluster=(1, 4), resolved=(1, 3), fallback=False)
    lams = np.array([10.0, 20.0, 20.0, 20.5, 31.0])
    c = _case_record(SimpleNamespace(model=model, lams=lams, delta=0.1, lin=None, sub_ok=True))
    assert c['row'] == 'middle' and not c['certified']
    assert (c['m'], c['j'], c['resolved_m'], c['resolved_j']) == (3, 1, 2, 1)
    assert c['split'] == 0.5 and c['gap_below'] == 10.0 and c['gap_above'] == 10.5
    assert c['n_active_constraints'] == 0 and c['delta'] == 0.1


def test_the_record_has_no_gap_past_the_ends_of_the_computed_spectrum():
    model = dict(mode='cluster', m=2, j=0, cluster=(0, 2), resolved=(0, 2), fallback=False)
    c = _case_record(SimpleNamespace(model=model, lams=np.array([20.0, 20.5]), delta=0.1,
                                     lin=None, sub_ok=True))
    assert np.isnan(c['gap_below']) and np.isnan(c['gap_above'])


def test_a_run_cut_off_after_a_step_says_its_case_is_stale():
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=3, warn_gap=0.0, degenerate_tol=0.0)
    p0 = param_vector(_perturbed(4, 0.06), PERIM)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = maximize_ratio(ev, p0, k=1, max_iter=1)
    assert not np.array_equal(res.x, p0), 'the one iteration should have moved'
    assert res.final_case is not None and res.final_case['at_endpoint'] is False


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('crit', [4.6e-6, 1e-7])
def test_the_sdp_step_finds_a_small_criticality_exactly(seed, crit):
    """A full-rank dual `Y` makes the optimum a closed form, `||w(Y)||`, and `crit` is built in.
    Left to cvxpy's default solver the step here had negative gain (rc1 row 0, 2026-11-10)."""
    rng = np.random.default_rng(seed)
    m, n, lam_1, lam_k = 3, 8, 28.0, 144.0
    Ms = rng.normal(size=(n, m, m))*60.0
    Ms = (Ms + Ms.transpose(0, 2, 1))/2
    Q = np.linalg.qr(rng.normal(size=(m, m)))[0]
    Y = (Q*np.array([0.1, 0.3, 0.6])) @ Q.T
    # `w` orthogonal to every trace-free direction of `Y`, so that this `Y` is the minimizer.
    E = [np.outer(np.eye(m)[a], np.eye(m)[b]) for a, b in zip(*np.triu_indices(m))]
    free = [e + e.T - 2*np.trace(e)*np.eye(m)/m for e in E]
    A = np.array([[np.sum(Mi*E) for E in free] for Mi in Ms])
    w = np.linalg.svd(A, full_matrices=True)[0][:, -1]*crit
    g1 = (np.einsum('ijk,jk->i', Ms, Y)*lam_1 - lam_1**2*w)/lam_k
    model = dict(mode='cluster', m=m, j=0, Ms=Ms, g1=g1, Lam=np.zeros((m, m)), lam_1=lam_1,
                 lam_k=lam_k)
    delta = 1.5e-6
    d = _step_concave_sdp(model, delta)
    assert model_gain(model, d)/delta == pytest.approx(crit, abs=1e-9)
