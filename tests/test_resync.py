"""rho never compares two discretizations.  (2026-11-08)

`rho = (F(q) - F(p))/gain` is a difference of two solves, and it is meaningful well below the
solves' absolute error only because both share a plan, so their errors correlate. A trial that
re-plans puts `q` on a new plan and leaves `p` on the old one. Each path here is one scripted
fake away from a minutes-long optimization.
"""
import numpy as np

from douse.cluster_opt import (Outcome, _degeneracy_move, _Opts, _repair_discontinuity, _State,
                               _trial_step)
from douse.errors import UnresolvedSpectrum
from douse.optimize import gauss_newton


class Fam:
    perim = 4.0

    def is_valid(self, p):
        return True


class PlanEv:
    """`eigenvalues` plays a script of `(lams, plan)` or an exception; a refusal leaves the
    plan alone, as the real evaluator guarantees."""

    def __init__(self, script, plan='A'):
        self.script, self.current_plan, self.family = list(script), plan, Fam()
        self.calls, self.reseeded, self.restored = [], [], []
        self.degenerate_tol, self.n_replans = 0.0, 0

    def eigenvalues(self, p, windows=None):
        self.calls.append(np.array(p, float))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        lams, self.current_plan = item
        return np.asarray(lams, float), f'solved on {self.current_plan}'

    def plan_state(self):
        return self.current_plan

    def restore_plan(self, state):
        self.restored.append(state)
        self.current_plan = state

    def reseed(self, p, lams, solved=None):
        self.reseeded.append(np.array(p, float))

    def reset_continuation(self):
        pass


def opts():
    return _Opts(k=1, delta0=0.05, delta_max=1.0, eps_rel='auto', auto=True, max_iter=10,
                 gain_tol=1e-7, eta=0.1, eta_grow=0.75, plateau_rtol=None, plateau_shrink=1e-3,
                 plateau_max=1e-3, callback=None, rng=None, verify_final=False, constraints=None,
                 degeneracy_moves=False, quasi_newton=False, max_unresolved_streak=5,
                 discontinuity_halvings=4, discontinuity_ratio=2.0, sigma_abort=None,
                 sigma_abort_streak=3, fam=Fam())


def state(delta=0.4, gain=1.0):
    st = _State(p=np.zeros(2), F=2.0, lams=np.array([1.0, 2.0]), solver='solved on A', delta=delta)
    st.begin_iteration(1)
    st.d, st.gain = np.array([0.1, 0.0]), gain
    return st


def test_with_no_replan_a_trial_costs_one_solve_and_rho_is_against_the_carried_value():
    st, ev = state(), PlanEv([([1.0, 2.5], 'A')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    assert len(ev.calls) == 1 and st.rho == 0.5 and st.F == 2.5


def test_a_replanning_trial_is_compared_with_the_iterate_on_the_new_plan():
    # on plan B the iterate reads 2.3, so the trial at 2.5 gained 0.2, not 0.5
    st, ev = state(), PlanEv([([1.0, 2.5], 'B'), ([1.0, 2.3], 'B')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.CONTINUE
    assert np.array_equal(ev.calls[1], np.zeros(2))
    assert abs(st.rho - 0.2) < 1e-12 and st.F == 2.5 and st.solver == 'solved on B'
    assert np.allclose(st.p, [0.1, 0.0]), 'accepted'
    assert np.allclose(ev.reseeded[-1], [0.1, 0.0]), 'the continuation goes on from the trial'
    assert st.n_resyncs == 1


def test_a_rejected_replanning_trial_leaves_the_iterate_on_the_new_plan():
    """The old code kept (p, F) on the old plan after a rejection, while the evaluator had moved
    on, so the NEXT trial crossed plans as well."""
    st, ev = state(), PlanEv([([1.0, 2.1], 'B'), ([1.0, 2.3], 'B')])
    _trial_step(st, opts(), ev, {})
    assert abs(st.rho + 0.2) < 1e-12 and st.delta == 0.2
    assert np.array_equal(st.p, np.zeros(2)) and st.F == 2.3 and st.solver == 'solved on B'
    assert np.array_equal(ev.reseeded[-1], np.zeros(2))


def test_when_the_iterate_refuses_the_new_plan_the_old_one_comes_back_and_the_trial_is_refused():
    st, ev = state(), PlanEv([([1.0, 2.5], 'B'), UnresolvedSpectrum('not on this plan')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.NEXT_ITER
    assert ev.current_plan == 'A' and st.F == 2.0 and st.solver == 'solved on A'
    assert st.delta == 0.2 and st.n_unresolved == 1 and st.refusal_streak == 1


def test_when_the_iterate_replans_in_turn_the_trial_is_void_and_only_a_repeat_shrinks():
    st, ev = state(), PlanEv([([1.0, 2.5], 'B'), ([1.0, 2.2], 'C')])
    assert _trial_step(st, opts(), ev, {}) is Outcome.NEXT_ITER
    assert np.array_equal(st.p, np.zeros(2)) and st.F == 2.2 and st.solver == 'solved on C'
    assert st.delta == 0.4
    ev.script = [([1.0, 2.5], 'D'), ([1.0, 2.2], 'E')]
    assert _trial_step(st, opts(), ev, {}) is Outcome.NEXT_ITER
    assert st.delta == 0.2, 'ping-pong is bounded'


def test_a_swap_clears_the_rejections_it_would_otherwise_misread():
    st, ev = state(), PlanEv([([1.0, 2.1], 'B'), ([1.0, 2.3], 'B')])
    st.reject_run = [(0.8, -1e-3), (0.4, -2e-3)]
    _trial_step(st, opts(), ev, {})
    assert len(st.reject_run) == 1, 'only the rejection on the new plan'


def test_a_repair_that_replans_is_adopted_even_without_a_jump_in_F():
    st, ev = state(delta=0.0125), PlanEv([([1.0, 2.0], 'B')])
    st.reject_run = [(0.1, 1e-3), (0.05, 1e-3), (0.025, 1e-3), (0.0125, 1e-3)]
    _repair_discontinuity(st, opts(), ev, {'n': 0, 'seen': set()})
    assert st.solver == 'solved on B' and st.n_discontinuity_repairs == 0


# ---- the degeneracy move ---------------------------------------------------------------------

def test_a_declined_degeneracy_move_puts_the_runs_plan_back(monkeypatch):
    from douse import cluster_opt as co

    class DM(PlanEv):
        def __init__(self):
            super().__init__([([1.0, 1.5], 'X')], plan='A')
            self.family = Fam()
            self.family.vertices = lambda p: np.array([0, 1, 1 + 1j, 1j])

        def replan(self, p=None):
            self.current_plan = None
    monkeypatch.setattr(co._dg, 'flatten_candidates', lambda v, perim: [v])
    monkeypatch.setattr(co._dg, 'collapse_candidates', lambda v, perim: [])
    monkeypatch.setattr(co, 'gauge_vector', lambda v, perim, anchor: np.zeros(2))
    ev = DM()
    assert _degeneracy_move(ev, np.zeros(2), 2.0, 1) is None
    assert ev.current_plan == 'A'


def test_an_accepted_degeneracy_move_keeps_the_winners_own_plan(monkeypatch):
    from douse import cluster_opt as co

    class DM(PlanEv):
        def __init__(self):
            super().__init__([([1.0, 3.0], 'W'), ([1.0, 1.5], 'L')], plan='A')
            self.family = Fam()
            self.family.vertices = lambda p: np.array([0, 1, 1 + 1j, 1j])

        def replan(self, p=None):
            self.current_plan = None
    v = np.array([0, 1, 1 + 1j, 1j])
    monkeypatch.setattr(co._dg, 'flatten_candidates', lambda z, perim: [v, v])
    monkeypatch.setattr(co._dg, 'collapse_candidates', lambda z, perim: [])
    monkeypatch.setattr(co, 'gauge_vector', lambda z, perim, anchor: np.zeros(2))
    ev = DM()
    best = _degeneracy_move(ev, np.zeros(2), 2.0, 1)
    assert best[1] == 3.0 and ev.current_plan == 'W'


# ---- gauss_newton ----------------------------------------------------------------------------

class Match:
    n_eigs = 1

    def residual(self, lams, J):
        return np.asarray(lams, float), np.asarray(J, float)


class GNEv:
    family, n_eigs = Fam(), 1

    def __init__(self, script):
        self.script, self.current_plan, self.seen, self.restored = list(script), 'A', [], []

    def plan_state(self):
        return self.current_plan

    def restore_plan(self, state):
        self.restored.append(state)
        self.current_plan = state

    def __call__(self, q):
        self.seen.append(np.array(q, float))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        value, self.current_plan = item
        return np.array([value]), np.array([[1.0, 0.0]])


def test_gauss_newton_compares_a_replanned_trial_with_the_start_on_the_new_plan():
    # 0.9 beats the carried 1.0 but not the start's own 0.8 on plan B: rejected
    ev = GNEv([(1.0, 'A'), (0.9, 'B'), (0.8, 'B'), (0.7, 'B')])
    res = gauss_newton(ev, Match(), np.zeros(2), max_iter=1, max_step=0.5)
    assert len(ev.seen) == 4 and np.array_equal(ev.seen[2], np.zeros(2))
    assert np.isclose(res.fun, 0.49)


def test_gauss_newton_restores_the_plan_when_the_start_refuses_it():
    ev = GNEv([(1.0, 'A'), (0.5, 'B'), UnresolvedSpectrum('no'), (0.9, 'A')])
    res = gauss_newton(ev, Match(), np.zeros(2), max_iter=1, max_step=0.5)
    assert ev.restored == ['A'] and res.n_refused == 1 and np.isclose(res.fun, 0.81)
