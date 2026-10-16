"""The 2026-10-15 stall-analysis tools (`benchmarks/diagnostics/`) on the fake rectangle.  (2026-10-16)

Each tool's core runs here on `FakeRectEvaluator` from the golden-trace test (closed-form
rectangle spectrum, no solver noise), so the plumbing is checked in seconds; what the tools
measure on the real solver is in `docs/stall_analysis_2026_10_15.md`, and a `slow` smoke test
at the end runs the two that only mean something there.
"""
import json

import numpy as np
import pytest

import douse.cluster_opt as CO
from douse import ClosingBudget, ConstraintSet, MinEdgeLength, VertexPolygon, param_vector

from benchmarks.diagnostics import (endpoint, fd_hessian, instrumented_run, probe_stall, replay,
                                    scan_step, warm_grad)
from test_cluster_opt_trace import FakeRectEvaluator

V_RECT = np.array([0, 1.25, 1.25 + 0.75j, 0.75j]) + np.array([0.03 + 0.02j, -0.02 + 0.01j,
                                                               0.01 - 0.03j, 0.02j])


def _setup():
    fam = VertexPolygon(4, perim=4.0)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    return fam, FakeRectEvaluator(fam), cons, param_vector(V_RECT, 4.0)


def test_probe_rho_is_near_one_where_the_objective_is_smooth():
    fam, ev, cons, p = _setup()
    F0, recs = probe_stall.probe(fam, ev, cons, p, k=1, radii=(1e-3, 1e-4, 1e-5))
    assert F0 == pytest.approx(endpoint.ratio(ev.eigenvalues(p)[0], 1))
    assert [r['delta'] for r in recs] == [1e-3, 1e-4, 1e-5]
    for r in recs:
        assert r['mode'] == 'smooth' and r['feasible']
        assert r['slope'] > 1e-2                     # far from the optimum L = H
        assert abs(r['rho'] - 1) < 0.05              # first order is exact to O(delta)


def test_scan_fit_matches_the_model_slope():
    fam, ev, cons, p = _setup()
    r = scan_step.scan(fam, ev, cons, p, k=1)
    assert len(r['samples']) == len(scan_step.TS)
    assert r['slope'] == pytest.approx(r['model_slope'], rel=1e-2)
    assert r['rms'] < 1e-8


def test_fd_hessian_is_symmetric_on_the_fake():
    fam, ev, cons, p = _setup()
    out = fd_hessian.fd_hessian(fam, ev, p, k=1, h=1e-4)
    for name in ('H1', 'Hmu'):
        assert out[name]['H'].shape == (len(p), len(p))
        assert out[name]['norm'] > 0
        assert out[name]['asymmetry'] < 1e-4


def test_warm_grad_agrees_where_warm_and_cold_are_the_same():
    fam, _, _, p = _setup()
    r = warm_grad.compare(fam, lambda: FakeRectEvaluator(fam), p, k=1)
    assert r['lam_windowed'] == 0 and r['lam_prefix'] == 0
    for key in ('g1_windowed', 'g1_prefix', 'gmu_windowed', 'gmu_prefix'):
        assert r[key]['rel'] < 1e-12


@pytest.mark.parametrize('qn', ['scalar', 'sr1', 'none'])
def test_replay_records_each_iterate(qn):
    fam, ev, cons, p = _setup()
    res, hist = replay.replay(endpoint.warm_path(ev, 'prefix'), p, 1, cons, max_iter=3,
                              quasi_newton=replay.QN[qn], rng=np.random.default_rng(0))
    assert 1 <= len(hist) <= 3 + 1
    assert hist[-1]['F'] == pytest.approx(res['fun'], rel=1e-9) or res['nit'] < 3
    assert all(set(h) == {'t', 'F', 'mode', 'cluster', 'delta', 'rho', 'split'} for h in hist)


def test_instrumented_run_logs_and_restores_the_stages(tmp_path):
    fam, ev, cons, p = _setup()
    stock = CO._STAGES
    out = tmp_path / 'log.jsonl'
    with open(out, 'w') as fh:
        res, _ = instrumented_run.run(ev, p, 1, cons, max_iter=4, quasi_newton='scalar',
                                      sink=instrumented_run._JsonlSink(fh),
                                      rng=np.random.default_rng(0))
    assert CO._STAGES is stock
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs
    for r in recs:
        assert {'it', 'F', 'mode', 'cluster', 'delta', 'gain1', 'gain2', 'H1', 'Hmu',
                'outcome2'} <= set(r)
        instrumented_run.show_line(r)                # the --show formatter reads every record
    assert any('gamma' in (r['H1'] or {}) for r in recs)


def test_instrumented_restores_stages_on_error():
    stock = CO._STAGES
    with pytest.raises(RuntimeError):
        with instrumented_run.instrumented([]):
            assert CO._STAGES is not stock
            raise RuntimeError
    assert CO._STAGES is stock


def test_show_reads_the_committed_sr1_log(capsys):
    import os
    path = os.path.join(os.path.dirname(endpoint.OVERNIGHT), 'sr1_blowup_seed0_prefix_2026_10_15.jsonl')
    instrumented_run.show(path)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 223 and 'H1[' in lines[0]


def test_load_row_reads_the_overnight_ledger():
    row = endpoint.load_row(2, 'windowed2')
    assert (row['N'], row['k'], len(row['x'])) == (6, 2, 8)
    with pytest.raises(LookupError):
        endpoint.load_row(99, 'nonesuch')


def test_warm_path_names():
    ev = object()
    assert endpoint.warm_path(ev, 'windowed') is ev
    assert isinstance(endpoint.warm_path(ev, 'prefix'), endpoint.PrefixEvaluator)
    with pytest.raises(ValueError):
        endpoint.warm_path(ev, 'cold')


@pytest.mark.slow
def test_real_solver_smoke():
    """`fd_hessian` and `warm_grad` on the real solver, at a rectangle (N = 4, k = 1)."""
    from douse import runs
    fam = runs.family_for(4)
    p = param_vector(V_RECT, runs.PERIM)
    ev = runs.make_evaluator(4, 1, fam=fam)
    out = fd_hessian.fd_hessian(fam, ev, p, 1, h=1e-4)
    assert out['H1']['asymmetry'] < 1e-4
    r = warm_grad.compare(fam, lambda: runs.make_evaluator(4, 1, fam=fam), p, 1)
    assert r['g1_windowed']['rel'] < 1e-8 and r['g1_prefix']['rel'] < 1e-8
