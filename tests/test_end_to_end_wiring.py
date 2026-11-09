"""The optimizer's options, on a real solve, reaching the solver.  (2026-11-08)

Fakes pass whatever the optimizer hands them, so the audit's mutants that unwired `_error_floor`,
`prec_final` and the tracked accuracy survived every test. This runs the real stack once, on the
triangle (lam_2/lam_1, maximized by the equilateral at 7/3), and watches what crosses the seam.
"""
import inspect

import numpy as np
import pytest

from chladni.eigenproblem import Eigenproblem
from douse import SpectrumEvaluator, VertexPolygon, cluster_opt as co, param_vector
from douse import chladni_backend as cb

PERIM = 3.0
TAU_G = 2.5e-6
LADDER = [1e-12, 1e-11, 1e-10]


@pytest.fixture(scope='module')
def run():
    seen = dict(prec=[], windows=[], tau_g=[], floors=[])
    solve, partition, floor = Eigenproblem.solve, co.partition_levels, co._error_floor

    def spy_solve(self, *a, **kw):
        seen['prec'].append(kw.get('prec'))
        if kw.get('windows') is not None:
            seen['windows'].append((a[0], len(kw['windows'])))
        return solve(self, *a, **kw)

    def spy_partition(spec, tau):
        seen['tau_g'].append(tau)
        return partition(spec, tau)

    def spy_floor(st, o):
        out = floor(st, o)
        seen['floors'].append(out)
        return out
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Eigenproblem, 'solve', spy_solve)
        mp.setattr(co, 'partition_levels', spy_partition)
        mp.setattr(co, '_error_floor', spy_floor)
        fam = VertexPolygon(3, perim=PERIM)
        z = np.exp(2j*np.pi*np.arange(3)/3)*(1 + 0.01*np.array([1, -1, 0.5]))
        p0 = param_vector(z*PERIM/np.abs(np.roll(z, -1) - z).sum(), PERIM)
        ev = SpectrumEvaluator(fam, n_eigs=4, prec=1e-10, degenerate_tol=0.0)
        res = co.maximize_ratio(ev, p0, k=1, max_iter=60, delta0=0.01, prec_final=LADDER,
                                tau_g=TAU_G, rng=np.random.default_rng(0))
    return res, seen


@pytest.mark.slow
def test_the_run_reaches_the_known_answer_and_certifies(run):
    res, _ = run
    assert res.fun == pytest.approx(7/3, abs=1e-6)
    assert res.success and res.verified


@pytest.mark.slow
def test_the_final_precision_reaches_the_solver_and_the_result_says_so(run):
    res, seen = run
    assert any(p in LADDER for p in seen['prec'])
    assert res.prec_final_reached


@pytest.mark.slow
def test_the_gradient_accuracy_reaches_the_level_partition(run):
    _, seen = run
    assert seen['tau_g'] and set(seen['tau_g']) == {TAU_G}


@pytest.mark.slow
def test_warm_solves_get_one_window_per_requested_value(run):
    _, seen = run
    assert seen['windows'] and all(n == want for want, n in seen['windows'])


@pytest.mark.slow
def test_the_error_floor_is_measured_and_recorded(run):
    res, seen = run
    assert seen['floors'] and np.isfinite(res.eps_f) and res.eps_f > 0
    assert np.isfinite(res.eps_g) and res.eps_g > 0


@pytest.mark.parametrize('name,kw', [
    ('solve', {'prec', 'plan', 'near', 'windows'})])
def test_chladni_still_takes_what_douse_passes(name, kw):
    params = inspect.signature(getattr(Eigenproblem, name)).parameters
    assert kw <= set(params)


def test_the_fakes_answer_to_the_real_evaluator_signature():
    """A fake with a narrower signature hides an argument the optimizer started passing."""
    from test_cluster_opt_stages import FakeEvaluator
    from test_cluster_opt_trace import FakeRectEvaluator
    real = set(inspect.signature(cb.SpectrumEvaluator.eigenvalues).parameters)
    for fake in (FakeEvaluator, FakeRectEvaluator):
        assert real <= set(inspect.signature(fake.eigenvalues).parameters), fake.__name__
