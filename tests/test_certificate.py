"""What a run may claim when it stops.  (2026-11-08)

The final audit found the claims running ahead of the evidence: a semidefinite subproblem posed in
absolute units returned zero gain at small radii, so `crit = 0` read as "first-order optimal" at a
shape with an ascent direction (N = 6, lam_6/lam_1: 0 at radius 7.6e-6 against 0.061 true); and
the plateau test, which cannot fail on a smooth model, was certifying at crit up to 1e-3.
"""
import numpy as np
import pytest

from douse.cluster_opt import (_Opts, _State, _floor_stop, _stationary, _subproblem, _verify_final,
                               trust_region_step)
from douse.errors import SolverRefusal


def _block(m, n_p, rng):
    A = rng.normal(size=(n_p, m, m))
    return (A + np.transpose(A, (0, 2, 1)))/2


def _cluster_model(m, j, n_p=12, seed=0, lam_1=18.0, lam_k=110.0):
    rng = np.random.default_rng(seed)
    return dict(mode='cluster', m=m, j=j, Ms=_block(m, n_p, rng), g1=0.1*rng.normal(size=n_p),
                Lam=np.zeros((m, m)), lam_1=lam_1, lam_k=lam_k, cluster=(5, 5 + m), scale=1.0)


@pytest.mark.parametrize('m,j', [(2, 0), (3, 0), (4, 0), (3, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize('delta', [1e-4, 1e-6, 1e-8, 1e-10])
def test_a_small_radius_does_not_change_the_criticality(m, j, delta):
    """The model is homogeneous when `Lam = 0`, so `gain/delta` is the same at every radius."""
    model = _cluster_model(m, j)
    rng = np.random.default_rng(0)
    _, g_ref = trust_region_step(model, 1e-2, rng=rng)
    _, g = trust_region_step(model, delta, rng=rng)
    assert g_ref > 0
    assert g/delta == pytest.approx(g_ref/1e-2, rel=2e-2)


def test_a_failed_semidefinite_solve_falls_back_and_is_not_read_as_optimal(monkeypatch):
    import douse.cluster_opt as co
    monkeypatch.setattr(co, '_step_concave_sdp', lambda model, delta: None)
    d, g, ok = _subproblem(_cluster_model(3, 0), 1e-6, rng=np.random.default_rng(0))
    assert ok and g > 0


def test_nothing_usable_from_any_solver_cannot_certify(monkeypatch):
    import douse.cluster_opt as co
    monkeypatch.setattr(co, '_step_concave_sdp', lambda model, delta: None)
    monkeypatch.setattr(co, '_unit_ball_search', lambda model, delta, seeds, lin=None:
                        np.zeros(len(seeds[0])))
    _, g, ok = _subproblem(_cluster_model(3, 0), 1e-6, rng=np.random.default_rng(0))
    assert g == 0 and not ok


class _Fam:
    def vertices(self, p):
        return np.array([0, 1, 1 + 1j, 1j])


def _opts(**kw):
    base = dict(k=1, delta0=0.05, delta_max=0.5, eps_rel='auto', auto=True, max_iter=10,
                gain_tol=1e-7, eta=0.1, eta_grow=0.75, plateau_rtol=0.05, plateau_shrink=1e-3,
                plateau_max=1e-3, callback=None, rng=np.random.default_rng(0), verify_final=True,
                constraints=None, degeneracy_moves=True, quasi_newton=False,
                max_unresolved_streak=5, discontinuity_halvings=4, discontinuity_ratio=2.0,
                sigma_abort=1e-6, sigma_abort_streak=3, fam=_Fam())
    base.update(kw)
    return _Opts(**base)


def _state(crit, eps_g, delta=1e-3, model=None):
    st = _State(p=np.zeros(4), F=2.0, lams=np.array([10.0, 20.0, 30.0]), solver=object(),
                delta=delta)
    st.begin_iteration(1)
    st.model = dict(mode='smooth', grad=np.array([crit, 0, 0, 0]), g1=np.zeros(4), cluster=(1, 2)) \
        if model is None else model
    st.crit, st.gain, st.gain1, st.eps_g, st.eps_f = crit, crit*delta, crit*delta, eps_g, 1e-12
    return st


def test_criticality_above_the_derivatives_accuracy_does_not_stop():
    """A smooth model plateaus at any criticality; only the accuracy of the gradient may stop it."""
    assert _stationary(_state(9e-4, 1e-8), _opts()) is None


def test_criticality_within_the_derivatives_accuracy_stops_and_says_so():
    res, message = _stationary(_state(5e-8, 1e-8), _opts())
    assert res == pytest.approx(5e-8)
    assert 'eps_g' in message


def test_inaccurate_derivatives_never_certify_past_the_cap():
    assert _stationary(_state(5e-3, 1e-3), _opts()) is None


def test_without_a_measured_gradient_error_the_plateau_stays():
    assert _stationary(_state(1e-5, np.nan), _opts()) is not None


def test_a_zero_that_a_larger_radius_contradicts_is_not_a_certificate():
    """A concave model's criticality cannot grow with the radius, so positive gain at a larger
    radius proves the zero at this one came from the subproblem."""
    st = _state(0.0, 1e-8, delta=1e-7, model=_cluster_model(3, 0))
    assert _stationary(st, _opts()) is None
    assert st.crit > 0.01


def test_a_failed_subproblem_cannot_stop():
    st = _state(0.0, 1e-8)
    st.sub_ok = False
    assert _stationary(st, _opts()) is None


def test_a_floor_ending_above_the_certificate_is_not_a_success():
    st = _state(9e-4, 1e-8)
    st.floor_run = 3
    assert _floor_stop(st, _opts())
    assert not st.success
    assert st.floor_reason == 'eps_f'
    assert 'not certified' in st.message
    assert st.resolution == pytest.approx(9e-4)


def test_a_floor_ending_within_the_certificate_is_a_success():
    st = _state(5e-8, 1e-8)
    st.floor_run = 3
    assert _floor_stop(st, _opts())
    assert st.success
    assert 'stationary to within eps_g' in st.message


class _Ev:
    def __init__(self, lams=None, refuse=False, prec=1e-12):
        self.lams, self.refuse, self.prec, self.calls = lams, refuse, prec, []

    def reset_continuation(self):
        pass

    def eigenvalues(self, p):
        if self.refuse:
            raise SolverRefusal('no')
        return self.lams, object()

    def set_prec(self, p, prec):
        self.calls.append(prec)
        return np.array([10.0, 20.0, 30.0]), object()


def _ended(lams=(10.0, 20.0, 30.0), eps_f=1e-10):
    st = _State(p=np.zeros(4), F=2.0, lams=np.array(lams), solver=object(), delta=1e-3)
    st.success, st.message, st.eps_f = True, 'first-order optimal', eps_f
    return st


def test_a_fresh_solve_that_agrees_verifies():
    st = _ended()
    _, _, verified = _verify_final(st, _opts(), _Ev(np.array([10.0, 20.0, 30.0])),
                                   dict(n=0, n_hadamard_short=0, seen=set()))
    assert verified and st.success


def test_a_fresh_solve_that_disagrees_is_not_verified():
    st = _ended()
    fresh = np.array([10.0, 20.001, 30.0])
    _, _, verified = _verify_final(st, _opts(), _Ev(fresh),
                                   dict(n=0, n_hadamard_short=0, seen=set()))
    assert not verified and not st.success
    assert st.F == pytest.approx(2.0001)
    assert 'disagrees' in st.message
    assert st.message.startswith('first-order optimal')


def test_a_refusal_at_the_final_precision_retries_at_the_starting_one():
    st = _ended()
    st.prec_final_reached, st.prec_start = True, 1e-9
    ev = _Ev(refuse=True)
    _, _, verified = _verify_final(st, _opts(), ev, dict(n=0, n_hadamard_short=0, seen=set()))
    assert verified and st.success
    assert ev.calls == [1e-9]


def test_an_unverifiable_endpoint_keeps_what_the_ending_said():
    st = _ended()
    _, _, verified = _verify_final(st, _opts(), _Ev(refuse=True),
                                   dict(n=0, n_hadamard_short=0, seen=set()))
    assert not verified and not st.success
    assert st.message.startswith('first-order optimal')
    assert 'could not be verified' in st.message


# ---- endings and the rarer paths ------------------------------------------------------------


def test_an_unsuccessful_ending_still_says_how_stationary_it_was():
    from douse.cluster_opt import _close_ending
    st = _ended()
    st.success, st.last_crit = False, 3e-4
    _close_ending(st)
    assert st.resolution == pytest.approx(3e-4) and st.floor_reason == 'max_iter'


class _PlanEv:
    """An evaluator whose plan moved after the previous iterate was solved."""
    def __init__(self):
        self.current_plan, self.calls = 'B', 0

    def eigenvalues(self, p):
        self.calls += 1
        return np.array([1.0, 3.0]), 'on B'

    def reseed(self, p, lams, solved=None):
        pass


def test_going_back_across_a_plan_re_evaluates_the_earlier_iterate():
    from douse.cluster_opt import _model_refused
    st = _State(p=np.ones(2), F=2.5, lams=np.array([1.0, 2.5]), solver='new', delta=1e-5)
    st.previous, st.model_plan = (np.zeros(2), 2.0, np.array([1.0, 2.0]), 'on A'), 'A'
    ev = _PlanEv()
    _model_refused(st, _opts(), ev, SolverRefusal('no'))
    assert ev.calls == 1 and st.solver == 'on B' and st.F == pytest.approx(3.0)


def test_going_back_on_the_same_plan_costs_no_solve():
    from douse.cluster_opt import _model_refused
    st = _State(p=np.ones(2), F=2.5, lams=np.array([1.0, 2.5]), solver='new', delta=1e-5)
    st.previous, st.model_plan = (np.zeros(2), 2.0, np.array([1.0, 2.0]), 'on B'), 'B'
    ev = _PlanEv()
    _model_refused(st, _opts(), ev, SolverRefusal('no'))
    assert ev.calls == 0 and st.solver == 'on B' and st.F == pytest.approx(2.0)


class _LadderEv:
    def __init__(self, deliver):
        self.prec, self.deliver, self.calls = 1e-9, deliver, []

    def set_prec(self, p, prec):
        self.calls.append(prec)
        if prec not in self.deliver:
            raise SolverRefusal('no')
        return np.array([1.0, 2.5]), 'solved'

    def reseed(self, p, lams, solved=None):
        pass


def _ladder(ev, v=(0, 1, 1 + 1j, 1j)):
    from douse.cluster_opt import _escalate

    class F:
        def vertices(self, p):
            return np.array(v, complex)
    st = _state(1e-9, 1e-8)
    return st, _escalate(st, _opts(prec_final=[1e-12, 1e-11, 1e-10], fam=F()), ev, True)


def test_a_ladder_keeps_the_first_rung_delivered():
    ev = _LadderEv({1e-11})
    st, redone = _ladder(ev)
    assert redone and ev.calls == [1e-12, 1e-11] and st.prec_final_reached


def test_a_nonconvex_iterate_starts_the_ladder_at_what_it_can_deliver():
    ev = _LadderEv({1e-11})
    _, redone = _ladder(ev, v=(0, 2, 0.5 + 0.5j, 2j))
    assert redone and ev.calls == [1e-11]


def test_a_small_step_refuses_rather_than_re_plans():
    from douse.cluster_opt import REPLAN_MIN_RADIUS, _trial_step

    class Ev:
        may_replan, seen = True, []
        degenerate_tol = 0.0

        def eigenvalues(self, p):
            self.seen.append(self.may_replan)
            raise SolverRefusal('no')

        def reseed(self, p, lams, solved=None):
            pass

    class F(_Fam):
        def is_valid(self, p):
            return True
    for delta, want in ((10*REPLAN_MIN_RADIUS, True), (0.1*REPLAN_MIN_RADIUS, False)):
        ev = Ev()
        ev.seen = []
        st = _state(1e-3, 1e-8, delta=delta)
        st.d = np.zeros(4)
        _trial_step(st, _opts(fam=F()), ev, dict(n=0, seen=set()))
        assert ev.seen == [want] and ev.may_replan
