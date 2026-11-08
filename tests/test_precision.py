"""Precision comes from the problem.  (2026-11-08)

A ratio run starts cheap and asks for 1e-12 once, when its step gains no more than the starting
precision can judge, and only on convex shapes (re-entrant corners floor near 1e-11). A fit's
precision follows the noise in its data.
"""
import numpy as np
import pytest

from douse.chladni_backend import SpectrumEvaluator
from douse.cluster_opt import Outcome, _first_order_step
from douse.errors import SolverRefusal
from douse.polygon import VertexPolygon, is_convex
from benchmarks.inverse_multistart import fit_prec
from test_resync import opts, state

import douse.cluster_opt as co

SQUARE = [0, 1, 1 + 1j, 1j]
DART = [0, 2, 0.5 + 0.5j, 2j]


def test_convexity():
    assert is_convex(SQUARE) and is_convex(SQUARE[::-1]) and not is_convex(DART)
    assert is_convex([0, 1, 2, 2 + 1j, 1j]), 'a straight vertex is not re-entrant'


class Fam:
    def __init__(self, v=SQUARE):
        self.v = v

    def is_valid(self, p):
        return True

    def vertices(self, p):
        return np.array(self.v, complex)


class Ev:
    def __init__(self, fail=False):
        self.prec, self.fail, self.calls = 1e-9, fail, []

    def set_prec(self, p, prec):
        self.calls.append(prec)
        if self.fail:
            raise SolverRefusal('no')
        self.prec = prec
        return np.array([1.0, 2.5]), 'solved at 1e-12'

    def reseed(self, p, lams, solved=None):
        pass


def setup(gain, monkeypatch, final=1e-12, v=SQUARE, **over):
    st = state(delta=1.0)
    st.model, st.lin = dict(mode='smooth'), None
    monkeypatch.setattr(co, 'trust_region_step', lambda m, d, rng=None, lin=None: (np.zeros(2), gain*d))
    o = opts()
    o = o.__class__(**{**o.__dict__, 'prec_final': final, 'fam': Fam(v), 'plateau_rtol': None, **over})
    return st, o


def test_no_escalation_while_the_gain_is_resolvable(monkeypatch):
    st, o = setup(1e-3, monkeypatch)
    ev = Ev()
    assert _first_order_step(st, o, ev, {}) is Outcome.CONTINUE and ev.calls == []


def test_escalates_once_when_the_gain_reaches_the_starting_precision(monkeypatch):
    st, o = setup(1e-9, monkeypatch)
    ev = Ev()
    assert _first_order_step(st, o, ev, {}) is Outcome.NEXT_ITER
    assert ev.calls == [1e-12] and st.F == 2.5 and st.solver == 'solved at 1e-12'
    assert st.prec_final_reached is True
    st.begin_iteration(2)
    st.model = dict(mode='smooth')
    assert _first_order_step(st, o, ev, {}) is Outcome.STOP and ev.calls == [1e-12], 'once'


def test_escalation_comes_before_a_stop_would_fire(monkeypatch):
    st, o = setup(1e-8, monkeypatch, gain_tol=1e-7)
    ev = Ev()
    assert _first_order_step(st, o, ev, {}) is Outcome.NEXT_ITER and not st.success


def test_a_nonconvex_iterate_stays_at_its_precision_and_may_stop(monkeypatch):
    st, o = setup(1e-8, monkeypatch, v=DART, gain_tol=1e-7)
    ev = Ev()
    assert _first_order_step(st, o, ev, {}) is Outcome.STOP
    assert ev.calls == [] and st.success and st.prec_final_reached is False


def test_undeliverable_precision_stays_put_and_says_so(monkeypatch):
    st, o = setup(1e-8, monkeypatch, gain_tol=1e-7)
    ev = Ev(fail=True)
    assert _first_order_step(st, o, ev, {}) is Outcome.STOP
    assert st.success and st.prec_final_reached is False and st.F == 2.0
    assert len(ev.calls) == 1


def test_off_by_default(monkeypatch):
    st, o = setup(1e-8, monkeypatch, final=None, gain_tol=1e-7)
    ev = Ev()
    assert _first_order_step(st, o, ev, {}) is Outcome.STOP and ev.calls == []
    assert st.prec_final_reached is None


def real():
    return SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-9)


def test_set_prec_restores_everything_on_failure(monkeypatch):
    ev = real()
    ev._plan, ev._plan_at, ev._last, ev._specs = 'plan', np.ones(4), ('p', 'l'), {'a': 1}
    def refuse(p, seed=None, windows=None):
        assert ev.prec == 1e-12 and ev._plan is None, 'replanned at the new precision'
        raise SolverRefusal('no')
    monkeypatch.setattr(ev, 'eigenvalues', refuse)
    n = ev.n_replans
    with pytest.raises(SolverRefusal):
        ev.set_prec(np.zeros(4), 1e-12)
    assert ev.prec == 1e-9 and ev._plan == 'plan' and ev._last == ('p', 'l') and ev._specs == {'a': 1}
    assert ev.n_replans == n


def test_set_prec_keeps_the_new_precision_on_success(monkeypatch):
    ev = real()
    monkeypatch.setattr(ev, 'eigenvalues', lambda p, seed=None, windows=None: ('lams', 'solved'))
    n = ev.n_replans
    assert ev.set_prec(np.zeros(4), 1e-12) == ('lams', 'solved')
    assert ev.prec == 1e-12 and ev.n_replans == n + 1


@pytest.mark.parametrize('sigma,want', [(0.0, 1e-9), (1e-8, 1e-9), (1e-6, 1e-8), (1e-7, 1e-9),
                                        (5e-7, 5e-9), (1e-3, 1e-8)])
def test_auto_precision_follows_the_noise(sigma, want):
    class S:
        prec = 'auto'
    assert fit_prec(S, sigma) == pytest.approx(want)


def test_an_explicit_precision_wins():
    class S:
        prec = 1e-10
    assert fit_prec(S, 1e-3) == 1e-10
