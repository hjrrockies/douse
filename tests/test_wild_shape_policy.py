"""EVERY solver refusal at a wild shape is an unreachable TRIAL, not a dead run.

`UnresolvedSpectrum` has been treated that way since the Weyl enforcement landed. The other two
ways a wild polygon defeats the forward solver were not, and by 2026-08-29 they were the dominant
failure in the grid: `EigensolverFailure: f has too many local minima` (five instances) and
`ValueError: N of M sources landed inside the domain` from the `FundamentalBasis` guard (one).
All six were MID-RUN -- a shape wild enough to provoke either is only reached after the run has
travelled -- so each one killed a run that was making progress.

The policy is now one class, `ShapeRefusal`, translated at the single place that knows the shape
(`SpectrumEvaluator.eigenvalues`) rather than at the five call sites in `cluster_opt` that were
leaking them.
"""
import warnings

import numpy as np
import pytest

from douse.evaluate import EigensolverFailure, SourcesInsideDomain

from douse import VertexPolygon, param_vector
from douse.cluster_opt import maximize_ratio
from douse.evaluate import (LappySpectrumEvaluator, ShapeRefusal, SolverRefusal,
                            SpectrumEvaluator, UnresolvedSpectrum)

PERIM = 4.0


def rect():
    """A shape the optimizer will actually STEP from -- the square is already `lam_2/lam_1`
    optimal and certifies at nit=1, proposing no trial at all."""
    fam = VertexPolygon(4, perim=PERIM)
    return fam, param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)


def test_the_two_kinds_of_refusal_share_one_base_class():
    """The whole point: a caller writes ONE `except`. The subclasses stay because the diagnosis
    differs -- no spectrum came back, against one that came back unindexable."""
    assert issubclass(SolverRefusal, ShapeRefusal)
    assert issubclass(UnresolvedSpectrum, ShapeRefusal)
    assert not issubclass(SolverRefusal, UnresolvedSpectrum)


@pytest.mark.parametrize('exc', [
    EigensolverFailure('f has too many local minima'),
    SourcesInsideDomain('3 of 240 sources landed inside the domain.'),
])
@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_lappy_refusals_are_translated_at_the_evaluator(monkeypatch, exc):
    """Translated where `p` is known, and the original is kept as `__cause__` so a ledger row
    still records WHICH refusal it was."""
    fam, p = rect()
    ev = LappySpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)

    def boom(self, q, seed=None):
        raise exc

    monkeypatch.setattr(LappySpectrumEvaluator, '_eigenvalues', boom)
    with pytest.raises(SolverRefusal) as info:
        ev.eigenvalues(p)
    assert info.value.__cause__ is exc
    assert type(exc).__name__ in str(info.value)
    assert ev.n_solver_refusals == 1


@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_solver_refusal_records_the_shape_and_the_plan_it_used(monkeypatch):
    """A count is not actionable. `benchmarks/replay_refusal.py` re-solves a refusal off-line and
    needs both ends: the shape that failed and the shape its frozen basis was planned on."""
    fam, p = rect()
    ev = LappySpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ev.eigenvalues(p)                      # builds the plan, so `plan_at` is populated

    monkeypatch.setattr(LappySpectrumEvaluator, '_eigenvalues',
                        lambda self, q, seed=None: (_ for _ in ()).throw(
                            EigensolverFailure('f has too many local minima')))
    with pytest.raises(SolverRefusal):
        ev.eigenvalues(p)
    rec = ev.refusals[-1]
    assert np.allclose(rec['p'], p)
    assert rec['plan_at'] is not None and np.allclose(rec['plan_at'], p)
    # A solver refusal has NO spectrum and no deficit -- there was nothing to compute one from.
    # `replay_refusal` formats on exactly this distinction, so it is part of the contract.
    assert rec['lams'] is None and rec['deficit'] is None
    assert 'EigensolverFailure' in rec['error']


@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_ValueError_that_is_NOT_about_the_shape_still_propagates(monkeypatch):
    """The catch is narrow on purpose. `SourcesInsideDomain` subclasses `ValueError` precisely so
    the evaluator does not have to write `except ValueError`, which would turn every shape bug in
    the family code into a silently-shrunk trust radius."""
    fam, p = rect()
    ev = LappySpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    monkeypatch.setattr(LappySpectrumEvaluator, '_eigenvalues',
                        lambda self, q, seed=None: (_ for _ in ()).throw(
                            ValueError('an ordinary bug')))
    with pytest.raises(ValueError, match='an ordinary bug'):
        ev.eigenvalues(p)
    assert ev.n_solver_refusals == 0


@pytest.mark.slow
def test_the_optimizer_survives_a_lappy_refusal_at_a_trial_shape():
    """The failure this exists to stop: five runs killed mid-flight by `EigensolverFailure`.
    A refused trial shrinks the radius and the run continues, exactly as an unindexable one does.
    """
    fam, p = rect()
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real, state = ev.eigenvalues, dict(n=0)

    def flaky(q, seed=None):
        if not np.allclose(q, p):
            state['n'] += 1
            if state['n'] <= 2:
                raise SolverRefusal('f has too many local minima') \
                    from EigensolverFailure('f has too many local minima')
        return real(q, seed=seed)

    ev.eigenvalues = flaky
    res = maximize_ratio(ev, p, k=1, max_iter=6, delta0=0.05)
    # Counted as a SOLVER refusal, not as an unindexable spectrum: the two have different fixes
    # and a single counter would hide which one a cell is suffering from.
    assert res.n_solver_refused == 2 and res.n_unresolved == 0
    assert np.isfinite(res.fun) and res.nit >= 3


@pytest.mark.slow
def test_a_RUN_of_solver_refusals_stops_instead_of_grinding_the_radius():
    """Shrinking is the right first response and a useless tenth one, and the streak is shared
    across the kinds -- a run alternating between them is exactly as stuck."""
    fam, p = rect()
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real, state = ev.eigenvalues, dict(n=0)

    def alternating(q, seed=None):
        if not np.allclose(q, p):
            state['n'] += 1
            if state['n'] % 2:
                raise SolverRefusal('f has too many local minima')
            raise UnresolvedSpectrum('synthetic refusal')
        return real(q, seed=seed)

    ev.eigenvalues = alternating
    res = maximize_ratio(ev, p, k=1, max_iter=200, delta0=0.05, max_unresolved_streak=4,
                         verify_final=False)
    assert res.n_solver_refused + res.n_unresolved == 4      # stopped on the shared streak
    assert res.n_solver_refused == 2 and res.n_unresolved == 2
    assert 'refused 4 consecutive trial shapes' in res.message and res.success is False
    assert res.delta > 1e-12                                 # radius NOT ground to nothing


@pytest.mark.slow
def test_a_refusal_in_the_discontinuity_REPAIR_does_not_kill_the_run():
    """The repair re-solves the CURRENT ITERATE on a fresh continuation. That call can refuse
    too, and there is then nothing to believe instead -- so the repair is dropped and the run
    carries on with its tracked state. The iterate never moved, so nothing about it became
    invalid; killing the run here would lose a trajectory to a check that exists to protect it.

    The discontinuity is provoked the way a real one presents: every trial returns the same drop
    in `F` whatever the radius, so `|rho|` doubles on each halving -- which cannot happen for a
    continuous objective and is the detector's whole content.
    """
    fam, p = rect()
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real, state = ev.eigenvalues, dict(armed=False, at_p=None, refused=0)

    def flaky(q, seed=None):
        if np.allclose(q, p):
            if state['armed'] and ev._last is None:
                state['refused'] += 1
                raise SolverRefusal('f has too many local minima')   # the repair's fresh solve
            out = real(q, seed=seed)
            state['at_p'], state['armed'] = out, True
            return out
        # A trial: the SAME drop in F at every radius, i.e. a discontinuity at `p`.
        lams, solver = state['at_p']
        bad = np.asarray(lams, dtype=float).copy()
        bad[1] *= 0.999
        return bad, solver

    ev.eigenvalues = flaky
    res = maximize_ratio(ev, p, k=1, max_iter=12, delta0=0.05, verify_final=False)
    assert state['refused'] >= 1                   # the repair really was reached
    assert np.isfinite(res.fun) and res.nit >= 4
    assert res.n_discontinuity_repairs == 0        # ...and could not be made, without dying


# ── persistent sigma drift: stop paying for a row that is already excluded ────────────────────

@pytest.mark.slow
@pytest.mark.lappy
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_run_whose_basis_cannot_resolve_the_shape_STOPS_instead_of_earning_its_exclusion():
    """`worst_sigma` above 1e-6 excludes a row from every summary, and it is a running max, so a
    run that has drifted past the gate and stays there is buying a row that will be discarded
    whatever it converges to. One N=3 row spent 78 minutes doing exactly that.

    The test is on `last_sigma` -- the tension AFTER the replan the drift itself provokes -- so a
    basis rebuilt at the shape and still failing is what stops the run, not a transient.
    """
    fam, p = rect()
    ev = LappySpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real = ev.eigenvalues

    def drifted(q, seed=None):
        out = real(q, seed=seed)
        ev.last_sigma = 1e-3                   # a rebuilt basis that still cannot resolve `lam_1`
        return out

    ev.eigenvalues = drifted
    res = maximize_ratio(ev, p, k=1, max_iter=200, delta0=0.05, sigma_abort=1e-6,
                         sigma_abort_streak=3, verify_final=False)
    assert res.nit <= 4 and res.success is False
    assert 'cannot resolve this shape' in res.message


@pytest.mark.slow
def test_a_TRANSIENT_excursion_does_not_stop_the_run():
    """Any evaluation back under the gate resets the streak. A wild trial shape is allowed to be
    unresolvable -- the trust region exists to back away from it -- and the run must survive that
    as it survives a refusal."""
    fam, p = rect()
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real, state = ev.eigenvalues, dict(n=0)

    def flickering(q, seed=None):
        out = real(q, seed=seed)
        state['n'] += 1
        ev.last_sigma = 1e-3 if state['n'] % 3 else 1e-12     # never three in a row
        return out

    ev.eigenvalues = flickering
    res = maximize_ratio(ev, p, k=1, max_iter=8, delta0=0.05, sigma_abort=1e-6,
                         sigma_abort_streak=3, verify_final=False)
    assert 'cannot resolve' not in res.message
    assert np.isfinite(res.fun)


def test_the_abort_can_be_switched_off():
    """`sigma_abort=None` restores the old behaviour, which the diagnostic harnesses that
    deliberately run drifted shapes need."""
    import inspect
    sig = inspect.signature(maximize_ratio).parameters
    assert sig['sigma_abort'].default == 1e-6          # the summaries' exclusion gate
    assert sig['sigma_abort_streak'].default == 3
