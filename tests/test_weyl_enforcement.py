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


@pytest.mark.slow
def test_a_refusal_records_BOTH_SHAPES_so_it_can_be_replayed(monkeypatch):
    """A count is not actionable: re-solving a refusal off-line needs the shape that failed AND
    the shape its frozen basis was planned on. `benchmarks/replay_refusal.py` consumes exactly
    this pair, and `lappy`'s reentrant sweep takes the same two ends."""
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    poison(monkeypatch, [9.0, 9.0])
    with pytest.raises(UnresolvedSpectrum):
        ev.eigenvalues(p)
    assert len(ev.refusals) == 1
    rec = ev.refusals[0]
    assert np.allclose(rec['p'], p)
    assert rec['plan_at'] is not None and np.allclose(rec['plan_at'], p)
    assert len(rec['lams']) == 3 and rec['deficit'] >= 9.0
    assert isinstance(rec['from_full_solve'], bool)


@pytest.mark.slow
def test_the_kept_refusals_are_BOUNDED(monkeypatch):
    """A 400-iterate start already has a memory problem; an unbounded list would join it."""
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0)
    ev.max_refusals_kept = 2
    monkeypatch.setattr('douse.evaluate.weyl_report',
                        lambda dom, lams, tol=0.5: (9.0, 'deficit 9.0'))
    for _ in range(5):
        with pytest.raises(UnresolvedSpectrum):
            ev.eigenvalues(p)
    assert ev.n_weyl_refusals == 5 and len(ev.refusals) == 2


def test_the_replay_tool_reads_a_row_shaped_like_the_ledger_writes_one():
    """The two ends of the pair, and the field names, are the contract between douse and lappy."""
    rr = pytest.importorskip('benchmarks.replay_refusal')
    import inspect
    src = inspect.getsource(rr.report)
    for field in ("'p'", "'plan_at'", "'deficit'"):
        assert field in src
    assert 'plan_at' in inspect.signature(rr.solve_at).parameters


@pytest.mark.slow
def test_a_refusal_in_verify_final_does_not_throw_away_the_run():
    """`verify_final` re-solves the endpoint. With enforcement on, that call can refuse -- and
    killing a converged run at its last step would lose the trajectory to a check added to
    protect it. The result comes back unverified and unsuccessful, saying so."""
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real, done = ev.eigenvalues, dict(final=False)

    def flaky(q, seed=None):
        if done['final']:
            raise UnresolvedSpectrum('synthetic refusal at the endpoint')
        return real(q, seed=seed)

    ev.eigenvalues = flaky

    def cb(*a):
        done['final'] = True                   # every call after the first iterate refuses

    res = maximize_ratio(ev, p, k=1, max_iter=1, delta0=0.05, callback=cb)
    assert res.verified is False and res.success is False
    assert 'could not be verified' in res.message
    assert np.isfinite(res.fun) and np.allclose(res.lams, res.lams_tracked)


@pytest.mark.slow
def test_a_RUN_of_refusals_stops_the_search_instead_of_grinding_the_radius():
    """Shrinking is the right first response and a useless tenth one: if the neighbourhood of
    this iterate cannot be indexed, no smaller step fixes it."""
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, degenerate_tol=0.0, warn_gap=0.0)
    real = ev.eigenvalues

    def always_refuse_trials(q, seed=None):
        if not np.allclose(q, p):
            raise UnresolvedSpectrum('synthetic refusal')
        return real(q, seed=seed)

    ev.eigenvalues = always_refuse_trials
    res = maximize_ratio(ev, p, k=1, max_iter=200, delta0=0.05, max_unresolved_streak=3,
                         verify_final=False)
    assert res.n_unresolved == 3               # stopped on the streak, not at max_iter
    assert 'could not be indexed' in res.message and res.success is False
    assert res.delta > 1e-12                   # the radius was NOT ground to nothing


def test_a_fatal_refusal_still_records_its_shapes():
    """A refusal at the START shape is fatal by design, but the shapes must survive the crash --
    they are what `replay_refusal` re-solves, and cell N6_k3 start 5 lost a pair this way."""
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    import inspect
    src = inspect.getsource(rm.run_one)
    assert 'except UnresolvedSpectrum' in src
    body = src[src.index('except UnresolvedSpectrum'):]
    for field in ('refusals=', "ok=False", 'trace='):
        assert field in body, field


@pytest.mark.slow
def test_the_spurious_bar_separates_a_real_eigenvalue_from_a_fake_one():
    """The constant `spurious_tol` rests on a measurement, and this pins it to the real case.

    The stalled re-run of N6_k1 start 2 carried 58.1948 as its `lam_3`. It is not an eigenvalue
    of that shape under any basis tried -- including the run's own -- and its tension says so by
    six orders of magnitude, while `worst_sigma` (measured at `lam_1`) stayed at 1.3e-09 and the
    Weyl deficit read 0.470 because a dropped mode plus a spurious one preserves the count.
    """
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'benchmarks', 'run', 'ratio_multistart', 'rerun', 'N6_k1_start2.jsonl')
    if not os.path.exists(path):
        pytest.skip('the stalled re-run row is not on disk')
    row = json.loads(open(path).read().strip())
    fam = VertexPolygon(6, perim=PERIM)
    p = np.array(row['x'], dtype=float)
    ev = SpectrumEvaluator(fam, n_eigs=row['k'] + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*row['k'] + 4), replan_tol=None, plan_at=p)
    lams, solver = ev.eigenvalues(p)
    sig = lambda x: float(np.atleast_1d(solver.sigma(float(x)))[0])                # noqa: E731
    genuine = max(sig(l) for l in lams)
    spurious = sig(58.1948)
    assert genuine < ev.spurious_tol < spurious
    assert spurious/genuine > 1e4                    # six orders in the measured case
    assert ev.worst_sigma_any >= ev.worst_sigma      # the first value is one of the values


@pytest.mark.slow
def test_worst_sigma_any_is_recorded_and_is_not_worst_sigma():
    """They answer different questions: `worst_sigma` is lam_1's resolution, which stays perfect
    while a higher slot holds garbage. Both are kept because the gates are calibrated on the
    first and only the second can see a spurious value."""
    fam = VertexPolygon(4, perim=PERIM)
    p = param_vector(np.array([0, 1.3, 1.3 + 0.7j, 0.7j], dtype=complex), PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=4, warn_gap=0.0, degenerate_tol=0.0, replan_tol=1e-9)
    lams, solver = ev.eigenvalues(p)
    per_value = [float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in lams]
    assert ev.worst_sigma == pytest.approx(per_value[0], rel=1e-9)
    assert ev.worst_sigma_any == pytest.approx(max(per_value), rel=1e-9)
    assert ev.worst_sigma_any >= ev.worst_sigma


def test_the_result_and_the_ledger_row_carry_the_plan_shape():
    """Two of the three candidate explanations for the stalled run needed the shape its plan was
    built at, and nothing had recorded it."""
    import inspect
    from douse import cluster_opt
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    assert 'plan_at=' in inspect.getsource(cluster_opt.maximize_ratio)
    assert 'plan_at=' in inspect.getsource(rm.run_one)
    assert 'worst_sigma_any=' in inspect.getsource(rm.run_one)


@pytest.mark.slow
def test_a_deficit_that_survives_a_solve_gets_a_REBUILT_BASIS_before_being_refused(monkeypatch):
    """The escalation re-solves on the SAME frozen plan, so a plan that cannot see a mode cannot
    see it the second time either. Replanning is the last rung: expensive and discontinuous, so
    it is tried only when a full solve has already failed."""
    fam, p = square()
    ev = SpectrumEvaluator(fam, n_eigs=3, degenerate_tol=0.0, warn_gap=0.0, replan_tol=1e-9)
    # A FRESH evaluator has no continuation, so its first reading already comes from a full
    # solve and the escalation rung is skipped: the ladder here is [solve bad] -> replan ->
    # [clean]. That asymmetry is the point of recording `from_full_solve` on a refusal.
    poison(monkeypatch, [9.0, 0.1])
    lams, _ = ev.eigenvalues(p)
    assert ev.n_weyl_replans == 1 and ev.n_weyl_recovered == 1 and ev.n_weyl_refusals == 0
    assert len(lams) == 3


@pytest.mark.slow
def test_the_recorded_N5_refusals_are_repaired_by_the_rebuild():
    """The evidence the rung was written from: all four refusals of N=5 `lam_6/lam_1` start 1 are
    the frozen plan returning ONE copy of a double `lam_1`. Sigma cannot see it -- the copy that
    is found sits at 5.8e-11 -- so only the count catches it, and only a rebuild repairs it."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks',
                        'run', 'ratio_multistart', 'rerun', 'N5_k5_contested_run3.jsonl')
    if not os.path.exists(path):
        pytest.skip('the contested re-run rows are not on disk')
    rows = [json.loads(l) for l in open(path)]
    ref = [r for r in rows if r['start'] == 1][0]['refusals'][0]
    fam = VertexPolygon(5, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=7, warn_gap=0.0, degenerate_tol=0.0, lam_max_window=14,
                           replan_tol=1e-9, plan_at=np.array(ref['plan_at'], dtype=float))
    lams, _ = ev.eigenvalues(np.array(ref['p'], dtype=float))
    assert ev.n_weyl_replans == 1 and ev.n_weyl_recovered == 1
    assert lams[0] == pytest.approx(lams[1], rel=1e-9), 'lam_1 is double there; both copies'


def test_a_redraw_does_not_inherit_the_discarded_start_s_statistics():
    """`worst_sigma` and friends are running maxima over an evaluator's lifetime, so reusing one
    across a redraw carries the DISCARDED shape's numbers into the run that replaces it.

    It cost a result: N=6 `lam_4/lam_1` start 5 discarded a reflex start whose tension was
    5.02e-04, converged from the redraw to 4.5353346506 -- the cell's answer to 5e-10, certified
    at criticality 6.7e-08 -- and was excluded on `worst_sigma = 5.0e-04`, a number belonging to a
    shape it never optimized. Its own endpoint tensions are 7.0e-13 to 3.3e-12.
    """
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    import inspect
    src = inspect.getsource(rm.run_one)
    body = src[src.index('for attempt in range'):]
    assert 'SpectrumEvaluator(' in body, 'the evaluator must be built INSIDE the redraw loop'
    before = src[:src.index('for attempt in range')]
    assert 'SpectrumEvaluator(' not in before, 'and not carried in from outside it'
