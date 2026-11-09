"""Steps along a curved bound, and what a run may claim there.  (2026-11-09)

Rows 9 and 11 of the lock-in validation stalled at criticality about 1 on the edge floor with a
correct gradient: every trial left the nonlinear bound at second order, and halving never helped.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import douse.cluster_opt as co
from douse.cluster_opt import (_convergence, _crit_tol, _second_order_correction, _stationary,
                               _subproblem)


def disk(p):
    """`1 - |p|^2 >= 0`: from the boundary, every tangent step leaves it at second order."""
    p = np.asarray(p, dtype=float)
    return np.array([1.0 - p @ p]), -2*p[None, :]


@pytest.mark.parametrize('delta', [1e-1, 1e-3, 1e-6])
def test_a_tangent_step_is_pulled_back_by_order_delta_squared(delta):
    p, d = np.array([1.0, 0.0]), np.array([0.0, delta])
    q = p + d
    assert disk(q)[0][0] < 0
    z = _second_order_correction(q, disk, delta)
    assert disk(z)[0][0] >= 0
    assert np.linalg.norm(z - q) == pytest.approx(delta**2/2, rel=0.1)


def test_a_feasible_trial_is_returned_untouched():
    q = np.array([0.5, 0.0])
    assert _second_order_correction(q, disk, 1e-3) is q


def test_a_correction_too_large_for_the_radius_is_declined():
    q = np.array([1.2, 0.0])
    assert _second_order_correction(q, disk, 1e-3) is q


def test_a_trial_the_constraints_cannot_evaluate_is_left_to_the_caller():
    def broken(p):
        raise ValueError('not a polygon')
    q = np.array([1.0, 1.0])
    assert _second_order_correction(q, broken, 1e-3) is q


class Disk:
    def __call__(self, p):
        return disk(p)

    def is_feasible(self, p, tol=0.0):
        return bool(np.all(disk(p)[0] >= -tol))


def _trial(monkeypatch, delta=1e-3):
    monkeypatch.setattr(co, '_warm_windows', lambda st, o: None)
    solved = []

    class Ev:
        def eigenvalues(self, q):
            solved.append(np.array(q))
            return np.array([1.0, 2.0 + 1e-3*delta]), 'solved'

    adopted = []
    st = SimpleNamespace(p=np.array([1.0, 0.0]), d=np.array([0.0, delta]), delta=delta, rho=0.0,
                         F=2.0, gain=1e-3*delta, n_infeasible=0, n_corrected=0, reject_run=[],
                         refusal_streak=0, resync_streak=0, floor_run=0, gain1=1e-3*delta,
                         eps_f=np.nan, adopt=lambda *a: adopted.append(a[0]),
                         reseed=lambda ev: None)
    o = SimpleNamespace(fam=SimpleNamespace(is_valid=lambda q: True), constraints=Disk(), k=1,
                        eta=0.1, eta_grow=0.75, delta_max=1.0)
    return st, o, Ev(), solved, adopted


def test_the_trial_step_solves_the_corrected_shape_instead_of_halving(monkeypatch):
    st, o, ev, solved, adopted = _trial(monkeypatch)
    co._trial_step(st, o, ev, None)
    assert st.n_infeasible == 0 and st.n_corrected == 1
    assert len(solved) == 1 and disk(solved[0])[0][0] >= 0
    assert np.array_equal(st.p + st.d, solved[0]) or np.allclose(st.p + st.d, solved[0], atol=1e-15)


def _smooth(n=2):
    return dict(mode='smooth', grad=np.eye(n)[0], m=1, j=0)


def _no_search(monkeypatch):
    monkeypatch.setattr(co, '_unit_ball_search',
                        lambda model, delta, seeds, lin=None: np.zeros(len(seeds[0])))


def test_a_smooth_constrained_step_does_not_depend_on_the_multistart(monkeypatch):
    """Ascent along the bound exists. Before this the search's empty hand read as a true zero."""
    _no_search(monkeypatch)
    lin = (np.array([0.0]), np.array([[-1.0, 1.0]]))         # stay on the side y >= x
    model = _smooth()
    d, g, ok = _subproblem(model, 1e-4, lin=lin)
    assert ok and model['exact_constrained'] and not model['fallback']
    assert g/1e-4 == pytest.approx(np.sqrt(0.5), rel=1e-4)


def test_a_smooth_constrained_true_maximum_is_a_real_zero():
    lin = (np.array([0.0]), np.array([[-1.0, 0.0]]))         # stay on the side x <= 0
    model = _smooth()
    d, g, ok = _subproblem(model, 1e-4, lin=lin)
    assert ok and g == 0 and model['exact_constrained']


def test_a_failed_conic_solve_with_an_empty_search_cannot_certify(monkeypatch):
    _no_search(monkeypatch)
    monkeypatch.setattr(co, '_step_smooth_constrained', lambda model, delta, lin: None)
    lin = (np.array([0.0]), np.array([[-1.0, 1.0]]))
    model = _smooth()
    _, g, ok = _subproblem(model, 1e-4, lin=lin)
    assert g == 0 and not ok and model['fallback']


def _cluster(j):
    Ms = np.zeros((2, 2, 2))
    Ms[0] = np.diag([1.0, 2.0])
    Ms[1] = np.diag([3.0, 1.0])
    return dict(mode='cluster', m=2, j=j, Ms=Ms, g1=np.zeros(2), Lam=np.zeros((2, 2)),
                lam_1=1.0, lam_k=2.0, cluster=(1, 3), scale=1.0)


def test_the_lowest_constrained_case_uses_the_conic_solve(monkeypatch):
    _no_search(monkeypatch)
    lin = (np.array([0.0]), np.array([[-1.0, 0.0]]))         # d[0] <= 0
    model = _cluster(0)
    d, g, ok = _subproblem(model, 1e-3, lin=lin)
    assert ok and model['exact_constrained'] and g > 0 and d[0] <= 1e-9


def test_the_highest_constrained_case_has_no_exact_method(monkeypatch):
    """Multistart only, so a zero stays possible and is reported as carrying no guarantee."""
    _no_search(monkeypatch)
    lin = (np.array([0.0]), np.array([[-1.0, 0.0]]))
    model = _cluster(1)
    _, g, ok = _subproblem(model, 1e-3, lin=lin)
    assert g == 0 and ok and not model['exact_constrained'] and not model['fallback']


def test_a_real_solve_without_a_measured_gradient_error_certifies_nothing():
    st = SimpleNamespace(sub_ok=True, eps_g=np.nan, crit=0.0, solver=SimpleNamespace(spec=object()))
    o = SimpleNamespace(gain_tol=1e-7, plateau_rtol=1e-3, plateau_max=1e-3)
    assert _stationary(st, o) is None
    assert 0.0 > _crit_tol(st, o)


def test_a_backend_that_measures_nothing_keeps_the_older_test():
    st = SimpleNamespace(sub_ok=True, eps_g=np.nan, crit=0.0, delta=1e-3, gain=0.0, solver=None)
    o = SimpleNamespace(gain_tol=1e-7, plateau_rtol=1e-3, plateau_max=1e-3)
    assert _stationary(st, o) is not None
    assert _crit_tol(st, o) == 1e-3


@pytest.mark.parametrize('success,case,state', [
    (False, dict(certified=True, at_endpoint=True), 'not_converged'),
    (True, dict(certified=True, at_endpoint=True), 'certified_stationary'),
    (True, dict(certified=False, at_endpoint=True), 'stationary_no_guarantee'),
    (True, dict(certified=True, at_endpoint=False), 'stationary_no_guarantee'),
    (True, None, 'stationary_no_guarantee'),
])
def test_the_three_states(success, case, state):
    assert _convergence(success, case) == state
