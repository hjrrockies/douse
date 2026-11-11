"""A run's record (`douse.record`): what it keeps, and that keeping it changes nothing.
(2026-11-11)

The recorder is diagnostics for runs in flight and after the fact, so it must never be the
reason a run went differently. The fake backend of the golden trace replays a run with and
without one and compares every callback and the whole result exactly; a two-iteration real run
on a rectangle checks the evaluator's lines (levels, bounds, solves, plans). A real 12-iteration
N=6 run was also compared byte for byte against the commit before the recorder (commit message).
"""
import json
from types import SimpleNamespace

import numpy as np

from douse import ClosingBudget, ConstraintSet, MinEdgeLength, SpectrumEvaluator, VertexPolygon
from douse import param_vector
from douse.cluster_opt import maximize_ratio
from douse.record import RunRecorder, read

from tests.test_cluster_opt_trace import FakeRectEvaluator, _jsonable


def _fake_run(recorder=None, max_iter=30):
    fam = VertexPolygon(4, perim=4.0)
    ev = FakeRectEvaluator(fam)
    cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
    v0 = np.array([0, 1.25, 1.25 + 0.75j, 0.75j]) + np.array(
        [0.03 + 0.02j, -0.02 + 0.01j, 0.01 - 0.03j, 0.02j])
    trace, models = [], []

    def cb(p, F, model, delta, rho):
        trace.append((list(map(float, p)), float(F), float(delta), float(rho)))
        models.append(model)
    kw = {} if recorder is None else dict(recorder=recorder)
    res = maximize_ratio(ev, param_vector(v0, 4.0), k=1, max_iter=max_iter, constraints=cons,
                         rng=np.random.default_rng(0), callback=cb, **kw)
    return trace, {k: _jsonable(v) for k, v in res.items()}, models, ev


def test_a_recorded_run_is_the_same_run(tmp_path):
    plain = _fake_run()
    with RunRecorder(tmp_path/'r.jsonl', dict(case='fake')) as rec:
        recorded = _fake_run(rec)
    assert recorded[0] == plain[0]                                  # every callback, exactly
    assert json.dumps(recorded[1], sort_keys=True) == json.dumps(plain[1], sort_keys=True)
    assert recorded[3].n_calls == plain[3].n_calls                  # no extra evaluations
    assert rec.n_errors == 0


def test_the_record_holds_the_model_the_step_and_the_ending(tmp_path):
    path = tmp_path/'r.jsonl'
    with RunRecorder(path, dict(case='fake')) as rec:
        trace, result, models, ev = _fake_run(rec)
    r = read(path)
    assert r['head'][0]['case'] == 'fake' and r['head'][0]['schema'] == 1
    assert len(r['iter']) == result['nit'] and len(r['result']) == 1
    assert r['result'][0]['fun'] == result['fun']
    assert [i['it'] for i in r['iter']] == list(range(1, result['nit'] + 1))
    # The fake's run crosses into cluster mode at the square's double: the record keeps the
    # cluster block, which is the whole first-order information of the nonsmooth F there.
    cl = [i for i in r['iter'] if i['model'] and i['model']['mode'] == 'cluster']
    assert cl, [i['model'] and i['model']['mode'] for i in r['iter']]
    for i in cl:
        m, n_p = i['model']['m'], len(i['p_model'])
        assert np.asarray(i['model']['Ms']).shape == (n_p, m, m)
        assert len(i['model']['Lam']) == m and len(i['model']['g1']) == n_p
    sm = [i for i in r['iter'] if i['model'] and i['model']['mode'] == 'smooth']
    assert sm and all(len(i['model']['grad']) == len(i['p_model']) for i in sm)
    # The record's model is the one the callback saw, array for array.
    by_model = [m for m in models if m is not None]
    stored = [i['model'] for i in r['iter'] if i['model'] is not None]
    for got, want in zip(stored, by_model):
        assert got['cluster'] == list(want['cluster'])
        np.testing.assert_array_equal(got['g1'], want['g1'])
        if want['mode'] == 'cluster':
            np.testing.assert_array_equal(got['Ms'], np.asarray(want['Ms']))
    # Step and decision: an accepted step moved the iterate by d; radii chain from line to line.
    for a, b in zip(r['iter'], r['iter'][1:]):
        assert b['delta_in'] == a['delta']
    acc = [i for i in r['iter'] if i['moved'] and i['d'] is not None]
    assert acc
    for i in acc:
        np.testing.assert_allclose(np.asarray(i['p']) - np.asarray(i['p_model']), i['d'],
                                   atol=1e-12)
    g = r['iter'][-1]['geometry']
    assert {'min_angle', 'max_angle', 'min_edge', 'slack', 'active'} <= set(g)
    assert set(g['slack']) == {'MinEdgeLength', 'ClosingBudget'}


def test_a_failing_recorder_does_not_touch_the_run(tmp_path):
    plain = _fake_run()
    rec = RunRecorder(tmp_path/'r.jsonl')
    rec._fh.close()                       # every write now fails
    broken = _fake_run(rec)
    assert broken[0] == plain[0]
    assert json.dumps(broken[1], sort_keys=True) == json.dumps(plain[1], sort_keys=True)
    assert rec.n_errors > 0


def test_read_skips_a_line_cut_off_by_a_killed_task(tmp_path):
    path = tmp_path/'r.jsonl'
    with RunRecorder(path, dict(case='x')) as rec:
        rec._write(dict(kind='iter', it=1))
    with open(path, 'a') as fh:
        fh.write('{"kind": "iter", "it": 2, "p": [0.1, ')
    r = read(path)
    assert [i['it'] for i in r['iter']] == [1]


def test_the_evaluation_line_reads_the_solve_record_and_its_spectrum(tmp_path):
    plan = SimpleNamespace(rung=2)
    level = SimpleNamespace(gap=0.5, cluster=None)
    value = SimpleNamespace(lam=np.array([10.0, 20.0]), mult=np.array([1, 2]),
                            bound=np.array([1e-9, 2e-9]), levels=(level, level), flags=('f',),
                            count_audit=SimpleNamespace(verdict='ok', statistic=0.1))
    solve = SimpleNamespace(spectrum=value, plan=plan, warm=True, warm_note='tracked')
    ev = SimpleNamespace(prec=1e-9, n_replans=0, last_sigma=2e-10,
                         family=VertexPolygon(4, perim=4.0))
    path = tmp_path/'r.jsonl'
    with RunRecorder(path) as rec:
        rec.solve('warm', None, plan, solve, 1.5)
        rec.evaluation(ev, param_vector(np.array([0, 1, 1 + 1j, 1j]), 4.0), spec=solve)
        rec.evaluation(ev, np.zeros(4), error='SolverRefusal: no')
    r = read(path)
    assert len(r['plan']) == 1 and r['plan'][0]['index'] == 0
    e, f = r['eval']
    assert e['plan'] == 0 and e['solves'][0]['plan'] == 0 and e['solves'][0]['kind'] == 'warm'
    assert e['lam'] == [10.0, 20.0] and e['mult'] == [1, 2] and e['warm'] is True
    assert e['audit'] == dict(verdict='ok', statistic=0.1) and e['sigma'] == 2e-10
    assert f['error'] == 'SolverRefusal: no' and f['solves'] == []


def test_two_real_iterations_leave_plans_solves_and_levels(tmp_path):
    """The evaluator's side on a real chladni solve (the 8 s pipeline test, recorded)."""
    L, H = 1.2, 0.8
    fam = VertexPolygon(4, perim=2*(L + H))
    path = tmp_path/'r.jsonl'
    with RunRecorder(path) as rec:
        ev = SpectrumEvaluator(fam, n_eigs=3, prec=1e-10, degenerate_tol=0.0, warn_gap=0.0)
        ev.recorder = rec
        cons = ConstraintSet(MinEdgeLength(fam), ClosingBudget(fam))
        res = maximize_ratio(ev, param_vector(np.array([0, L, L + 1j*H, 1j*H]), fam.perim), k=1,
                             max_iter=2, constraints=cons, verify_final=False, recorder=rec)
    r = read(path)
    assert rec.n_errors == 0, r.get('error')
    assert len(r['plan']) >= 1 and r['plan'][0]['prec'] == 1e-10
    assert r['plan'][0]['fields']['rung'] == r['eval'][0]['solves'][0]['rung']
    assert len(r['iter']) == res.nit == 2
    kinds = [s['kind'] for e in r['eval'] for s in e['solves']]
    assert kinds[0] == 'plan' and 'warm' in kinds, kinds
    for e in r['eval']:
        assert e['plan'] is not None and len(e['lam']) == len(e['mult']) == len(e['bound'])
        assert sum(e['mult']) >= 3 and all(b < 1e-8 for b in e['bound'])
    first = r['eval'][0]
    np.testing.assert_allclose(sorted(first['lam'])[0],
                               np.pi**2*(1/L**2 + 1/H**2), rtol=1e-9)
    assert r['iter'][0]['hadamard_attained']['lam_1'] < 1e-6
    assert np.isfinite(r['iter'][0]['eps_g']) and np.isfinite(r['iter'][0]['eps_f'])
