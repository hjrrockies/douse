"""The stages of `maximize_ratio`, each on a scripted fake.  (2026-10-14)

Until the loop was split these paths could only be reached by running the whole optimizer on a
real solve for minutes. Each test here builds a `_State` and `_Opts` by hand, drives one stage,
and checks the state it leaves: the refusal streak, the rho/radius update, the discontinuity
repair, the sigma-abort streak.
"""
import numpy as np
import pytest

from douse.cluster_opt import (Outcome, _Opts, _State, _repair_discontinuity, _sigma_abort,
                               _trial_step)
from douse.errors import SolverRefusal, UnresolvedSpectrum


class FakeFamily:
    def is_valid(self, p):
        return True

    def vertices(self, p):
        return np.array([0, 1, 1 + 1j, 1j])


class FakeEvaluator:
    """`eigenvalues` consumes a script: `('ok', lams)`, `('refuse', exc)`. Records what the
    optimizer writes back."""

    def __init__(self, script, last_sigma=np.nan):
        self.script = list(script)
        self.family = FakeFamily()
        self.degenerate_tol = 0.0
        self.n_replans = 0
        self.last_sigma = last_sigma
        self._last = None
        self.reseeds, self.resets, self.calls = [], 0, []

    def eigenvalues(self, p):
        self.calls.append(np.array(p, float))
        kind, payload = self.script.pop(0)
        if kind == 'refuse':
            raise payload
        return np.asarray(payload, float), object()

    def reset_continuation(self):
        self.resets += 1

    def __setattr__(self, name, value):
        if name == '_last' and value is not None:
            self.__dict__.setdefault('reseeds', []).append(value)
        object.__setattr__(self, name, value)


def _opts(**kw):
    base = dict(k=1, delta0=0.05, delta_max=0.5, eps_rel='auto', auto=True, max_iter=10,
                gain_tol=1e-7, eta=0.1, eta_grow=0.75, plateau_rtol=0.05, plateau_shrink=1e-3,
                plateau_max=1e-3, callback=None, rng=None, verify_final=False, constraints=None,
                degeneracy_moves=True, quasi_newton=False, max_unresolved_streak=5,
                discontinuity_halvings=4, discontinuity_ratio=2.0, sigma_abort=1e-6,
                sigma_abort_streak=3, fam=FakeFamily())
    base.update(kw)
    return _Opts(**base)


def _state(lams=(10.0, 20.0, 30.0), delta=0.05, gain=1.0):
    lams = np.array(lams, float)
    st = _State(p=np.zeros(4), F=float(lams[1]/lams[0]), lams=lams, solver=object(),
                delta=delta)
    st.begin_iteration(1)
    st.model, st.d, st.gain = dict(mode='smooth', cluster=(1, 2)), np.full(4, 0.01), gain
    return st


# ── the refusal streak ─────────────────────────────────────────────────────────────────────────

def test_a_refusal_shrinks_counts_reseeds_and_tries_again():
    ev = FakeEvaluator([('refuse', UnresolvedSpectrum('chladni: unmet'))])
    st, o, tally = _state(), _opts(), dict(n=0, seen=set())
    seen = []
    o = _opts(callback=lambda p, F, m, d, rho: seen.append((d, rho)))
    out = _trial_step(st, o, ev, tally)
    assert out is Outcome.NEXT_ITER
    assert st.delta == 0.025 and st.n_unresolved == 1 and st.n_solver_refused == 0
    assert st.refusal_streak == 1 and st.it == 1          # `nit` is the loop index, not +1
    assert len(ev.reseeds) == 1 and np.array_equal(ev.reseeds[0][1], st.lams)
    assert seen == [(0.025, -np.inf)]


def test_five_refusals_in_a_row_stop_the_run_with_the_last_kind_named():
    script = [('refuse', UnresolvedSpectrum('a'))]*3 + [('refuse', SolverRefusal('declined'))]*2
    ev = FakeEvaluator(script)
    st, o, tally = _state(), _opts(), dict(n=0, seen=set())
    outs = []
    for _ in range(5):
        outs.append(_trial_step(st, o, ev, tally))
    assert outs == [Outcome.NEXT_ITER]*4 + [Outcome.STOP]
    assert st.n_unresolved == 3 and st.n_solver_refused == 2 and st.refusal_streak == 5
    assert st.delta == pytest.approx(0.05/32)
    assert 'refused 5 consecutive trial shapes' in st.message and 'SolverRefusal' in st.message


def test_a_successful_trial_clears_the_streak():
    ev = FakeEvaluator([('refuse', UnresolvedSpectrum('a')), ('ok', [10.0, 25.0, 30.0])])
    st, o, tally = _state(), _opts(), dict(n=0, seen=set())
    _trial_step(st, o, ev, tally)
    st.begin_iteration(2); st.model, st.d, st.gain = dict(mode='smooth'), np.full(4, 0.01), 1.0
    assert _trial_step(st, o, ev, tally) is Outcome.CONTINUE
    assert st.refusal_streak == 0 and st.F == 2.5


# ── rho and the radius ─────────────────────────────────────────────────────────────────────────

def test_a_good_step_is_accepted_and_the_radius_grows_to_its_cap():
    ev = FakeEvaluator([('ok', [10.0, 29.0, 40.0])])            # F: 2 -> 2.9, gain 1 -> rho 0.9
    st, o, tally = _state(delta=0.4), _opts(), dict(n=0, seen=set())
    assert _trial_step(st, o, ev, tally) is Outcome.CONTINUE
    assert st.rho == pytest.approx(0.9) and st.F == pytest.approx(2.9)
    assert st.delta == 0.5 and st.reject_run == [] and not ev.reseeds


def test_a_modest_step_is_accepted_at_the_same_radius():
    ev = FakeEvaluator([('ok', [10.0, 25.0, 40.0])])            # rho 0.5: between eta and eta_grow
    st, o, tally = _state(delta=0.05), _opts(), dict(n=0, seen=set())
    _trial_step(st, o, ev, tally)
    assert st.rho == pytest.approx(0.5) and st.F == 2.5 and st.delta == 0.05


def test_a_bad_step_is_rejected_the_radius_halves_and_the_tracker_is_reseeded():
    ev = FakeEvaluator([('ok', [10.0, 20.5, 40.0])])            # rho 0.05 < eta
    st, o, tally = _state(delta=0.05), _opts(), dict(n=0, seen=set())
    _trial_step(st, o, ev, tally)
    assert st.F == 2.0 and st.delta == 0.025
    assert st.reject_run == [(0.05, pytest.approx(0.05))]
    assert len(ev.reseeds) == 1 and np.array_equal(ev.reseeds[0][0], st.p)


# ── the discontinuity repair ──────────────────────────────────────────────────────────────────

def _collapsing_run(delta0=0.05, n=4):
    """`|dF|` constant while `delta` halves: `|rho|` doubles every halving."""
    return [(delta0/2**i, -0.1) for i in range(n)]


def test_a_discontinuity_is_repaired_by_believing_the_fresh_solve():
    ev = FakeEvaluator([('ok', [10.0, 24.0, 40.0])])            # fresh F 2.4 vs carried 2.0
    st = _state(delta=0.05/16)
    st.reject_run = _collapsing_run()
    o, tally = _opts(), dict(n=0, seen=set())
    assert _repair_discontinuity(st, o, ev, tally) is Outcome.NEXT_ITER
    assert ev.resets == 1 and st.F == 2.4 and st.n_discontinuity_repairs == 1
    assert st.delta == 0.05 and st.reject_run == [] and st.it == 1


def test_a_refused_fresh_solve_drops_the_repair_and_keeps_the_run():
    ev = FakeEvaluator([('refuse', UnresolvedSpectrum('a'))])
    st = _state(delta=0.05/16)
    st.reject_run = _collapsing_run()
    o, tally = _opts(), dict(n=0, seen=set())
    assert _repair_discontinuity(st, o, ev, tally) is Outcome.CONTINUE
    assert st.F == 2.0 and st.n_discontinuity_repairs == 0 and st.reject_run == []
    assert st.delta == 0.05/16 and st.it == 1


def test_no_discontinuity_means_no_solve():
    ev = FakeEvaluator([])
    st = _state()
    st.reject_run = [(0.05, -0.1), (0.025, -0.05)]                # |rho| constant: continuous
    assert _repair_discontinuity(st, _opts(), ev, dict(n=0, seen=set())) is Outcome.CONTINUE
    assert ev.calls == []


# ── the sigma-abort streak ────────────────────────────────────────────────────────────────────

def test_the_sigma_streak_stops_on_the_third_bad_iterate_and_resets_on_a_good_one():
    ev = FakeEvaluator([], last_sigma=1e-5)
    st, o, tally = _state(), _opts(), dict(n=0, seen=set())
    assert _sigma_abort(st, o, ev, tally) is Outcome.CONTINUE and st.sigma_streak == 1
    ev.last_sigma = 1e-12
    assert _sigma_abort(st, o, ev, tally) is Outcome.CONTINUE and st.sigma_streak == 0
    ev.last_sigma = 1e-5
    for _ in range(2):
        assert _sigma_abort(st, o, ev, tally) is Outcome.CONTINUE
    assert _sigma_abort(st, o, ev, tally) is Outcome.STOP
    assert 'cannot resolve this shape' in st.message and not st.success


def test_no_statistic_means_no_abort():
    ev = FakeEvaluator([], last_sigma=np.nan)
    st = _state()
    for _ in range(5):
        assert _sigma_abort(st, _opts(), ev, dict(n=0, seen=set())) is Outcome.CONTINUE
    assert st.sigma_streak == 0
