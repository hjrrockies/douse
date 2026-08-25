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
"""
import warnings

import numpy as np
import pytest

from douse import VertexPolygon, param_vector, SpectrumEvaluator
from douse.cluster_opt import local_model, trust_region_step, maximize_ratio

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
def test_the_convex_case_is_refused_with_the_mathematics():
    """lam_3 is the HIGHEST member of its cluster at a regular N-gon, so its directional
    derivative is lam_max(M(d)) -- convex, and maximizing a convex function has no unique local
    model optimum. Refused rather than answered."""
    fam = VertexPolygon(4, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, warn_gap=0.0, degenerate_tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(NotImplementedError, match='convex rather than concave'):
            maximize_ratio(ev, param_vector(_regular(4), PERIM), k=2, max_iter=5)


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
