"""Every way a shape can defeat the forward solve is a rejected step, never a crash.  (2026-11-08)

The lock-in gate's non-convex inversions step near shapes the planner cannot serve, so a raw
exception from the planner, or from a Hadamard derivative at an accepted iterate, has to reach
the same "shrink and try again" handler a flagged spectrum does.
"""
import numpy as np
import pytest

from douse import SpectrumEvaluator, VertexPolygon
from douse.chladni_backend import ChladniSolver
from douse.cluster_opt import Outcome, _build_model, _Opts, _State
from douse.errors import DegenerateSpectrum, ShapeRefusal, SolverRefusal, UnresolvedSpectrum
from douse.optimize import gauss_newton
from douse.polygon import in_box

SQUARE = np.array([0, 1, 1 + 1j, 1j])



def test_a_planner_raise_is_a_solver_refusal():
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-8, degenerate_tol=0.0)
    ev._domain = lambda p: ('dom', SQUARE)

    def boom(p):
        raise ValueError('no source offset keeps the sources outside this domain')
    ev.plan = boom
    with pytest.raises(SolverRefusal, match='source offset'):
        ev.eigenvalues(np.zeros(4))


def test_a_runtime_error_from_the_geometry_is_a_solver_refusal():
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-8, degenerate_tol=0.0)

    def boom(p):
        raise RuntimeError('corner detection gave up')
    ev._domain = boom
    with pytest.raises(SolverRefusal):
        ev.eigenvalues(np.zeros(4))


def test_a_programming_error_is_not_dressed_up_as_a_refusal():
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-8, degenerate_tol=0.0)

    def boom(p):
        raise TypeError('wrong call')
    ev._domain = boom
    with pytest.raises(TypeError):
        ev.eigenvalues(np.zeros(4))


def test_a_failed_replan_leaves_the_previous_plan_in_place():
    """It used to leave `_plan_at` at the refused shape, so every later `plan()` retried there."""
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-8, degenerate_tol=0.0)
    ev._domain = lambda p: ('dom', SQUARE)
    good = object()
    ev._plan, ev._plan_at, ev._plan_built_at = good, None, np.ones(4)
    ev._solve = lambda dom, plan, p, near=None, windows=None: (None, UnresolvedSpectrum('drift'))

    def boom(p):
        raise ValueError('cannot plan here')
    original = SpectrumEvaluator.plan

    def plan(p):
        if ev._plan is None:
            boom(p)
        return original(ev, p)
    ev.plan = plan
    with pytest.raises(ShapeRefusal):
        ev.eigenvalues(np.zeros(4))
    assert ev._plan is good and ev._plan_at is None
    assert np.array_equal(ev._plan_built_at, np.ones(4))
    assert ev.current_plan is good


def test_a_replan_whose_spectrum_is_refused_restores_the_plan_too():
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, prec=1e-8, degenerate_tol=0.0)
    ev._domain = lambda p: ('dom', SQUARE)
    good, fresh = object(), object()
    ev._plan, ev._plan_built_at = good, np.ones(4)
    ev._solve = lambda dom, plan, p, near=None, windows=None: (None, UnresolvedSpectrum('no'))
    original = SpectrumEvaluator.plan

    def plan(p):
        if ev._plan is None:
            ev._plan, ev._plan_built_at = fresh, np.array(p, float)
        return original(ev, p)
    ev.plan = plan
    with pytest.raises(UnresolvedSpectrum):
        ev.eigenvalues(np.zeros(4))
    assert ev._plan is good


class _Spec:
    mult = np.array([1, 1])
    levels = (type('L', (), {'cluster': None})(), type('L', (), {'cluster': None})())


class _Record:
    spectrum = _Spec()

    def __init__(self, raises=None):
        self.raises, self.n = raises, 0

    def hadamard(self, i, V, report=False):
        self.n += 1
        if self.raises is not None:
            raise self.raises
        H = np.ones((2, 1, 1))
        return (H, 'the report') if report else H


def _solved(record):
    s = ChladniSolver(record, SQUARE, 0.0)
    s._velocity = lambda: None
    return s


def test_hadamard_keeps_its_report_and_costs_one_call():
    rec = _Record()
    s = _solved(rec)
    H = s.hadamard_vertices(0)
    assert H.shape == (2, 1, 1) and s.hadamard_vertices(0) is H
    assert rec.n == 1 and s.reports[0] == 'the report'


def test_a_hadamard_raise_is_a_solver_refusal():
    s = _solved(_Record(raises=ValueError('gram residual')))
    with pytest.raises(SolverRefusal, match='gram residual'):
        s.hadamard_vertices(0)


class _Fam:
    n = 4

    def is_valid(self, p):
        return True


class _ModelEv:
    family = _Fam()
    degenerate_tol = 0.0
    n_replans = 0
    _last = None

    def reseed(self, p, lams, solved=None):
        self.reseeded = np.array(p)


def _opts(streak=5):
    return _Opts(k=1, delta0=0.1, delta_max=1.0, eps_rel=1e-3, auto=False, max_iter=5,
                 gain_tol=1e-7, eta=0.1, eta_grow=0.75, plateau_rtol=None, plateau_shrink=1e-3,
                 plateau_max=1e-3, callback=None, rng=None, verify_final=False, constraints=None,
                 degeneracy_moves=False, quasi_newton=False, max_unresolved_streak=streak,
                 discontinuity_halvings=4, discontinuity_ratio=2.0, sigma_abort=None,
                 sigma_abort_streak=3, fam=_Fam())


class _Raises:
    """A solved shape whose derivative cannot be had."""
    def __init__(self, exc):
        self.exc = exc

    @property
    def spectrum(self):
        raise self.exc


def _state():
    st = _State(p=np.zeros(2), F=2.0, lams=np.array([1.0, 2.0]), solver=object(), delta=0.4)
    st.begin_iteration(1)
    return st


@pytest.mark.parametrize('exc,counter', [(SolverRefusal('h'), 'n_solver_refused'),
                                         (UnresolvedSpectrum('c'), 'n_unresolved'),
                                         (DegenerateSpectrum('d'), 'n_unresolved')])
def test_a_model_that_cannot_be_built_rolls_back_to_the_previous_iterate(exc, counter):
    st, ev = _state(), _ModelEv()
    previous = st.solver
    st.adopt(np.ones(2), 2.5, np.array([1.0, 2.5]), _Raises(exc))
    assert _build_model(st, _opts(), ev, {}) is Outcome.NEXT_ITER
    assert np.array_equal(st.p, np.zeros(2)) and st.F == 2.0 and st.solver is previous
    assert st.delta == 0.2 and getattr(st, counter) == 1 and st.refusal_streak == 1
    assert np.array_equal(ev.reseeded, np.zeros(2))


def test_a_model_refused_with_nowhere_to_go_back_to_stops_the_run_with_the_reason():
    st, ev = _state(), _ModelEv()
    st.solver = _Raises(SolverRefusal('no derivative here'))
    assert _build_model(st, _opts(), ev, {}) is Outcome.STOP
    assert 'no derivative here' in st.message and not st.success


def test_model_refusals_share_the_refusal_streak():
    st, ev = _state(), _ModelEv()
    st.refusal_streak = 1
    st.adopt(np.ones(2), 2.5, np.array([1.0, 2.5]), _Raises(SolverRefusal('h')))
    assert _build_model(st, _opts(streak=2), ev, {}) is Outcome.STOP


# ---- the box ------------------------------------------------------------------------------

GWW1 = np.array([1-3j, 3-1j, 3+1j, -1+1j, -1+3j, -3+1j, -1-1j, 1-1j])


@pytest.mark.parametrize('z', [SQUARE, GWW1, np.exp(2j*np.pi*np.arange(16)/16)])
def test_the_thesis_shapes_are_inside_the_box(z):
    assert in_box(z)


@pytest.mark.parametrize('z,why', [
    (np.array([0, 1, 0.5 + 0.02j]), 'angle'),                          # a 2-degree-ish apex
    (np.array([0, 1, 1 + 1j, 0.5 + 0.02j, 1j]), 're-entrant'),         # about 350 degrees
    (np.array([0, 1, 1 + 1j, 1.01 + 1j, 1.01 + 2j, 2j]), 'edge'),      # a 0.01 step
    (np.array([0, 3, 3 + 1j, 1.52 + 1j, 1.52 + 3j, 1.48 + 3j, 1.48 + 1j, 1j]), 'finger'),
    (np.array([0, 1, 0, 1j]) + np.array([0, 0, 1j, 1]), 'self-crossing'),
])
def test_shapes_outside_the_box_are_refused(z, why):
    assert not in_box(z), why


def test_in_box_does_not_care_about_orientation_or_scale():
    assert in_box(7.5 * GWW1[::-1])


# ---- gauss_newton --------------------------------------------------------------------------

class _Match:
    n_eigs = 1

    def residual(self, lams, J):
        return lams - 1.0, J


class _GNFam:
    @staticmethod
    def is_valid(p):
        return True


class _GNEv:
    family, n_eigs = _GNFam(), 1

    def __init__(self):
        self.seen = []

    def __call__(self, q):
        self.seen.append(np.array(q))
        return np.array([1.0 + float(q @ q)]), 2.0*np.atleast_2d(q)


def test_gauss_newton_spends_no_solve_on_a_trial_outside_the_admissible_set():
    ev = _GNEv()
    # the Gauss-Newton step halves q[0]: nothing below 0.9 is admissible, so every trial with
    # little damping is turned away before the evaluator sees it
    res = gauss_newton(ev, _Match(), np.array([1.0, 0.0]), max_iter=3, max_step=2.0,
                       admissible=lambda q: q[0] > 0.9)
    assert all(q[0] > 0.9 for q in ev.seen)
    assert res.n_outside > 0


def test_gauss_newton_without_a_box_is_unchanged():
    ev = _GNEv()
    res = gauss_newton(ev, _Match(), np.array([1.0, 0.0]), max_iter=3, max_step=2.0)
    assert res.n_outside == 0
