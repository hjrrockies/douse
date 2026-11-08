"""A run ends at an accuracy floor with the floor reported.  (2026-11-08)

Below the solves' error `rho` is noise, the radius halves to 1e-14, and the ledger said "trust
radius collapsed" (31 of 114 thesis_ratio_v2 rows). One rejection below the floor proves nothing:
same-plan errors correlate, so `dF` can be meaningful there. Three in a row is the signal.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from chladni.eigensolver import Cluster, Level, Spectrum
from scipy.optimize import OptimizeResult

from douse.chladni_backend import ChladniSolver
from douse.cluster_opt import Outcome, _error_floor, _finish_iteration, _second_order_step, _trial_step
from douse.ledger import result_row
from test_resync import PlanEv, opts, state


def below_floor(eps_f=1e-12):
    st = state(delta=1e-6, gain=1e-13)
    st.gain1, st.eps_f, st.eps_g, st.crit = 1e-13, eps_f, 1e-8, 1e-7
    return st


REJECT = ([1.0, 2.0 - 1e-13], 'A')


def reject_once(st, step=REJECT):
    ev = PlanEv([step])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    return _finish_iteration(st, opts(), ev, {})


def test_three_rejections_below_the_floor_end_the_run_with_the_floor_reported():
    st = below_floor()
    assert reject_once(st) is Outcome.CONTINUE and reject_once(st) is Outcome.CONTINUE
    assert reject_once(st) is Outcome.STOP
    assert 'stationary to within eps_g' in st.message and '(estimated)' in st.message
    assert st.success and st.floor_reason == 'eps_f' and st.resolution == st.crit


def test_success_needs_the_stationarity_measure_and_its_error_inside_the_plateau_gate():
    st = below_floor()
    st.eps_g = 1e-2                     # above plateau_max = 1e-3
    for _ in range(3):
        out = reject_once(st)
    assert out is Outcome.STOP and not st.success


def test_a_rejection_with_a_measurable_change_does_not_count():
    st = below_floor()
    reject_once(st), reject_once(st)
    reject_once(st, ([1.0, 2.0 - 1e-9], 'A'))
    assert st.floor_run == 0


def test_a_rejection_with_real_predicted_gain_does_not_count():
    st = below_floor()
    st.gain1 = 1e-9
    for _ in range(4):
        assert reject_once(st) is Outcome.CONTINUE
    assert st.floor_run == 0


def test_an_accepted_step_clears_the_run():
    st = below_floor()
    reject_once(st), reject_once(st)
    ev = PlanEv([([1.0, 2.0 + 1e-12], 'A')])
    _trial_step(st, opts(), ev, {})
    assert st.floor_run == 0


def test_a_backend_with_no_error_estimate_never_stops_at_the_floor():
    st = below_floor(eps_f=np.nan)
    for _ in range(5):
        assert reject_once(st) is Outcome.CONTINUE


def test_the_collapse_backstop_says_so():
    st = below_floor(eps_f=np.nan)
    st.delta = 1e-15
    assert _finish_iteration(st, opts(), PlanEv([]), {}) is Outcome.STOP
    assert st.floor_reason == 'delta_backstop' and 'collapsed' in st.message


def test_a_second_order_shrink_below_the_floor_counts_as_a_rejection():
    import douse.cluster_opt as co
    st = below_floor()
    st.floor_run = 2
    st.model = dict(mode='cluster')
    st.H1 = st.Hmu = SimpleNamespace(ready=False, B=None)
    o = opts()
    o = o.__class__(**{**o.__dict__, 'quasi_newton': True})
    keep = co._step_second_order, co.model_gain2
    co._step_second_order, co.model_gain2 = (lambda *a, **k: st.d), (lambda *a, **k: -1.0)
    try:
        assert _second_order_step(st, o, PlanEv([]), {}) is Outcome.STOP
    finally:
        co._step_second_order, co.model_gain2 = keep
    assert st.floor_reason == 'eps_f'


# --- the numbers themselves -----------------------------------------------------------------

class Fam:
    n = 3

    def d_vertices_dp(self, p):
        return 2.0*np.eye(6)


def rep(eps=1e-12, identity=1e-11, base=1e-13):
    return SimpleNamespace(sizing_prec=base, norm_rule_prec=base, gram_residual=(base,),
                           mp_eps=eps, identity_residual=identity, attained=np.nan)


def solved(levels, reports):
    sol = ChladniSolver(SimpleNamespace(spectrum=Spectrum(tuple(levels))), np.zeros(3, complex), 0.0)
    sol.reports = dict(reports)
    return sol


def test_a_near_pair_is_charged_for_the_gap_to_the_outside_not_the_gap_between_them():
    levels = [Level(10.0, 1, 1e-13, mp_eps=1e-12), Level(20.0, 1, 1e-13, mp_eps=1e-11),
              Level(20.0 + 1e-5, 1, 1e-13, mp_eps=1e-11), Level(40.0, 1, 1e-13, mp_eps=1e-12)]
    a = solved(levels, {(1, 2): rep(eps=1e-11)}).hadamard_attained(1, 3)
    x = 1e-11/0.5                          # lam_0 is half of lam_1: the relative gap, far side
    assert a == pytest.approx(3*max(1e-11, x*np.sqrt(1 + x*x)))


def test_a_level_with_no_residual_is_charged_only_its_identities():
    levels = [Level(10.0, 1, 1e-13), Level(20.0, 1, 1e-13)]
    a = solved(levels, {1: rep(eps=np.nan, identity=4e-11)}).hadamard_attained(1, 2)
    assert a == pytest.approx(1.2e-10)


def test_no_report_no_estimate():
    levels = [Level(10.0, 1, 1e-13, mp_eps=1e-12), Level(20.0, 1, 1e-13, mp_eps=1e-12)]
    assert np.isnan(solved(levels, {}).hadamard_attained(1, 2))


def test_the_floors_formula():
    levels = [Level(10.0, 1, 1e-10, mp_eps=1e-12), Level(20.0, 1, 4e-10, mp_eps=1e-12)]
    sol = solved(levels, {0: rep(eps=1e-12, identity=1e-11), 1: rep(eps=1e-12, identity=2e-11)})
    st = state()
    st.solver, st.F, st.lams = sol, 2.0, np.array([10.0, 20.0])
    st.model = dict(cluster=(1, 2), m=1)
    o = opts()
    o = o.__class__(**{**o.__dict__, 'fam': Fam()})
    eps_f, eps_g = _error_floor(st, o)
    assert eps_f == pytest.approx(2.0*(4e-10/20.0 + 1e-10/10.0))
    assert eps_g == pytest.approx(2.0*np.sqrt(6)*2.0*(3*2e-11 + 3*1e-11))


def test_a_fake_solver_has_no_floor():
    st = state()
    st.model = dict(cluster=(1, 2), m=1)
    assert np.isnan(_error_floor(st, opts())).all()


def test_the_ledger_carries_the_ending():
    res = OptimizeResult(fun=2.0, nit=1, x=np.zeros(2), lams=np.ones(2), success=True, message='m',
                         eps_f=1e-12, eps_g=np.nan, floor_reason='eps_f', prec_final_reached=None)
    row = result_row(res)
    assert row['eps_f'] == 1e-12 and row['eps_g'] is None
    assert row['floor_reason'] == 'eps_f' and row['prec_final_reached'] is None
