"""The Weyl deficit is ENFORCED, not merely reported.

Written for N=6 `lam_3/lam_1` start 3: at a reentrant angle of 284 degrees `track_set` returned
[56.90, 209.97, 274.11, 286.67] at a sigma of 4.7e-12, the optimizer accepted a step at
`rho = +1265` because `F` had leapt to a ratio of non-consecutive values, and then sat at a
frozen `F` for forty iterations while the radius collapsed. The deficit saw it (8.76) and the
run was not told.
"""
import numpy as np
import pytest

from douse import VertexPolygon, param_vector, spectral_guard
from douse.cluster_opt import maximize_ratio
from douse.evaluate import SpectrumEvaluator, UnresolvedSpectrum

PERIM = 4.0


def square():
    fam = VertexPolygon(4, perim=PERIM)
    return fam, param_vector(np.array([0, 1, 1 + 1j, 1j], dtype=complex), PERIM)


def poison(monkeypatch, values):
    """Make `weyl_report` return the given deficits in order, then 0.0 forever."""
    seq = list(values)

    def fake(dom, lams, tol=spectral_guard.DEFICIT_TOL):
        d = seq.pop(0) if seq else 0.0
        return d, (f'deficit {d}' if d > tol else '')

    monkeypatch.setattr('douse.evaluate.weyl_report', fake)


@pytest.mark.slow
def test_a_deficit_that_survives_a_full_solve_is_REFUSED(monkeypatch):
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    poison(monkeypatch, [9.0, 9.0])            # tracked set bad, and the full solve too
    with pytest.raises(UnresolvedSpectrum, match='not consecutive'):
        ev.eigenvalues(p)
    assert ev.n_weyl_refusals == 1
    assert ev.worst_weyl_deficit >= 9.0


@pytest.mark.slow
def test_the_FIRST_reading_is_never_acted_on_alone(monkeypatch):
    """A trip buys a full solve, and a clean full solve is accepted -- the deficit is a heuristic
    and the tracking seed is the usual culprit."""
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    ev.eigenvalues(p)                          # seed the continuation so tracking is live
    poison(monkeypatch, [9.0, 0.1])            # bad tracked set, clean on re-solve
    lams, _ = ev.eigenvalues(p*(1 + 1e-6))
    assert ev.n_weyl_escalations == 1 and ev.n_weyl_recovered == 1 and ev.n_weyl_refusals == 0
    assert np.all(np.diff(lams) >= 0) and lams[0] > 0


@pytest.mark.slow
def test_enforcement_can_be_turned_off_to_reproduce_the_old_behaviour(monkeypatch):
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0,
                          enforce_weyl=False)
    poison(monkeypatch, [9.0, 9.0])
    lams, _ = ev.eigenvalues(p)                # returned anyway, as before
    assert len(lams) == 3
    assert ev.worst_weyl_deficit >= 9.0 and ev.n_weyl_refusals == 0


@pytest.mark.slow
def test_the_optimizer_treats_a_refusal_as_an_unreachable_trial_and_keeps_going():
    """The point of the whole change: shrink and try a smaller step, rather than die or accept
    a ratio of non-consecutive values."""
    # A start the optimizer will actually STEP from: the square is already optimal for
    # `lam_2/lam_1`, so it certifies at nit=1 and no trial is ever proposed.
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real = ev.eigenvalues
    state = dict(n=0)

    def flaky(q, seed=None):
        # Refuse the first two TRIAL shapes only. A refusal at the current iterate is a different
        # matter and is deliberately left fatal: a run cannot start from a shape whose spectrum
        # it cannot index.
        if not np.allclose(q, p):
            state['n'] += 1
            if state['n'] <= 2:
                ev.n_weyl_refusals += 1
                raise UnresolvedSpectrum('synthetic refusal')
        return real(q, seed=seed)

    ev.eigenvalues = flaky
    res = maximize_ratio(ev, p, k=1, max_iter=6, delta0=0.05)
    assert res.n_unresolved == 2               # counted, and separate from n_infeasible
    assert np.isfinite(res.fun) and res.nit >= 3
