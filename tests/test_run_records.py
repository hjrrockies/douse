"""What a run leaves behind: which rows count, what a table may say, and surviving a Slurm limit.
(2026-11-08)

The final audit found tables keyed by start number across specs, a best-of-certified column that
hid better verified rows, resumes that kept another spec's rows, and day-long jobs that a limit
killed with nothing saved.
"""
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from douse import cluster_opt as co
from douse import runs
from douse.certify import antisymmetric_hessian
from douse.errors import ShapeRefusal, TimeBudget

SPECS = pathlib.Path(__file__).resolve().parent.parent/'benchmarks'/'specs'

sg = pytest.importorskip('benchmarks.summarize_grid')
rm = pytest.importorskip('benchmarks.ratio_multistart')
im = pytest.importorskip('benchmarks.inverse_multistart')


def _row(start, fun, success=False, sha='a', **kw):
    return dict(dict(N=6, k=4, start=start, fun=fun, success=success, spec=sha, spec_sha=sha, ok=True,
                     worst_sigma=1e-10, worst_bound=1e-10, worst_weyl_deficit=0.0, nit=5,
                     seconds=1.0, resolution=1e-3, message='m', verified=True), **kw)


def _ledger(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def test_a_start_number_means_something_only_within_its_spec(tmp_path):
    a = _ledger(tmp_path/'a.jsonl', [_row(1, 5.0, sha='a')])
    b = _ledger(tmp_path/'b.jsonl', [_row(1, 5.1, sha='b')])
    assert sorted(r['fun'] for r in sg.load_cell([a, b])) == [5.0, 5.1]


def test_the_best_found_is_reported_beside_a_lower_best_certified():
    rows = [_row(0, 5.0433, success=True, resolution=1e-7), _row(1, 5.0814)]
    s = sg.summarize(rows, 1e-6, 1e-4)
    assert s['best']['fun'] == pytest.approx(5.0433)
    assert s['best_found']['fun'] == pytest.approx(5.0814)


def test_an_unverified_row_is_not_the_best_found():
    s = sg.summarize([_row(0, 5.0), _row(1, 5.9, verified=False)], 1e-6, 1e-4)
    assert s['best_found']['fun'] == pytest.approx(5.0)


def test_the_endings_are_counted_by_reason():
    e = sg._endings([_row(0, 5.0, success=True), _row(1, 5.0, floor_reason='max_iter'),
                     _row(2, 5.0, floor_reason='max_iter', n_warm_refused=3, eps_g=2e-7)])
    assert e['reasons'] == {'certified': 1, 'max_iter': 2}
    assert e['eps_g'] == pytest.approx(2e-7) and e['n_warm_refused'] == 3


def test_a_cell_of_two_specs_is_refused_unless_asked_for(tmp_path, monkeypatch, capsys):
    (tmp_path/'N6_k4.jsonl').write_text(''.join(
        json.dumps(_row(i, 5.0, sha=s)) + '\n' for i, s in enumerate('ab')))
    monkeypatch.setattr('sys.argv', ['summarize_grid', '--dir', str(tmp_path)])
    with pytest.raises(SystemExit, match='mixes specs'):
        sg.main()
    monkeypatch.setattr('sys.argv', ['summarize_grid', '--dir', str(tmp_path), '--allow-mixed'])
    sg.main()
    assert 'how the rows ended' in capsys.readouterr().out


def test_an_inverse_resume_reruns_foreign_and_errored_rows(tmp_path):
    p = tmp_path/'t.jsonl'
    _ledger(p, [dict(id=0, spec_sha='s'), dict(id=1, spec_sha='old'),
                dict(id=2, spec_sha='s', status='error')])
    assert im._done_ids(p, 's') == {0}


def test_too_few_answered_probes_is_inconclusive():
    basis = np.eye(8)[:, :3]
    v = np.array([0, 1, 1 + 1j, 1j])
    calls = iter(range(1000))

    def F(x):
        i = next(calls)
        if i == 0:
            return 5.0
        if i % 4 != 1:
            raise ShapeRefusal('no')
        return 4.9
    out = antisymmetric_hessian(F, v, basis, ts=(0.01,))
    assert out['n_done'] < 0.75*out['n_probes'] and out['verdict'].startswith('inconclusive')


# ---- checkpoints and deadlines ---------------------------------------------------------------


@pytest.mark.parametrize('limit,seconds', [('24:00:00', 86400), ('3-00:00:00', 259200),
                                           ('00:30:15', 1815)])
def test_slurm_limits(limit, seconds):
    assert runs.slurm_seconds(limit) == seconds


def _tick(ck, n=1, F=2.0):
    for _ in range(n):
        ck(np.array([1.0, 2.0]), F, None, 0.25, 0.5)


def test_a_checkpoint_round_trips(tmp_path):
    p = tmp_path/'c.ckpt'
    ck = runs.Checkpoint(p, key='s', every=0.0)
    _tick(ck, 3)
    again = runs.Checkpoint(p, key='s')
    state = again.load()
    assert state['x'] == [1.0, 2.0] and state['delta'] == 0.25 and state['nit'] == 3
    assert again.nit == 3 and again.resumes == 1


def test_a_checkpoint_counts_on_across_a_resume(tmp_path):
    p = tmp_path/'c.ckpt'
    _tick(runs.Checkpoint(p, key='s', every=0.0), 3)
    second = runs.Checkpoint(p, key='s', every=0.0)
    second.load()
    _tick(second, 2)
    assert json.loads(p.read_text())['nit'] == 5 and json.loads(p.read_text())['resumes'] == 1


def test_another_specs_checkpoint_is_ignored(tmp_path):
    p = tmp_path/'c.ckpt'
    _tick(runs.Checkpoint(p, key='s', every=0.0))
    assert runs.Checkpoint(p, key='other').load() is None


def test_a_checkpoint_is_written_only_on_its_cadence(tmp_path):
    p = tmp_path/'c.ckpt'
    ck = runs.Checkpoint(p, key='s', every=3600.0)
    _tick(ck)
    assert not p.exists()
    ck.flush()
    assert p.exists() and not (tmp_path/'c.ckpt.tmp').exists()
    ck.clear()
    assert not p.exists()


def test_a_deadline_is_not_a_refusal():
    assert not issubclass(TimeBudget, ShapeRefusal)


def _clock(monkeypatch, expire_after):
    """`cluster_opt`'s clock, reaching the deadline at the given iteration check."""
    ticks = iter(range(10**6))
    monkeypatch.setattr(co, 'time', SimpleNamespace(
        time=lambda: 0.0 if next(ticks) < expire_after else 1.0))


@pytest.mark.slow
def test_running_out_of_time_saves_and_writes_no_row_and_resubmitting_continues(tmp_path, monkeypatch):
    p = tmp_path/'c.ckpt'
    args = (4, 1, 0, 40, 0, 0.05)
    _clock(monkeypatch, 3)
    ck = runs.Checkpoint(p, every=1e9)
    assert rm.run_one(*args, ckpt=ck, deadline=0.5) is None
    state = json.loads(p.read_text())
    assert state['nit'] >= 1 and state['prec'] == 1e-10

    monkeypatch.undo()
    again = runs.Checkpoint(p, every=1e9)
    row = rm.run_one(*args, ckpt=again, deadline=None)
    assert row['n_resumes'] == 1
    assert row['fun'] >= state['F'] - 1e-9
    fresh = rm.run_one(*args)
    assert fresh['n_resumes'] == 0
    assert row['fun'] == pytest.approx(fresh['fun'], rel=1e-5)


# ---- the runners hand the spec's options on ----------------------------------------------------


class _Stop(Exception):
    pass


def test_the_ratio_runner_hands_on_the_specs_final_precision(monkeypatch):
    import dataclasses
    from douse import spec as spec_mod
    seen = {}

    def spy(ev, p0, **kw):
        seen.update(kw)
        raise _Stop
    monkeypatch.setattr(rm, 'maximize_ratio', spy)
    spec = dataclasses.replace(spec_mod.load(SPECS/'thesis_ratio_v2.toml'),
                               prec_final=[1e-12, 1e-11])
    with pytest.raises(_Stop):
        rm.run_one(4, 1, 0, 5, 0, 0.05, spec=spec)
    assert seen['prec_final'] == [1e-12, 1e-11]


def test_an_ending_says_why_on_the_result():
    from douse import VertexPolygon, param_vector
    from test_cluster_opt_trace import FakeRectEvaluator
    fam = VertexPolygon(4, perim=4.0)
    p0 = param_vector(np.array([0, 1.25, 1.25 + 0.75j, 0.75j]), 4.0)
    res = co.maximize_ratio(FakeRectEvaluator(fam), p0, k=1, max_iter=1,
                            rng=np.random.default_rng(0), verify_final=False)
    assert res.floor_reason == 'max_iter'


def test_the_inverse_runner_hands_on_the_noise_target_and_the_box(monkeypatch):
    import dataclasses
    import douse
    seen = {}
    real = douse.SpectrumEvaluator.set_target

    def spy_target(self, target, sigma=0.0, tau_g=1e-6):
        seen['sigma'] = sigma
        return real(self, target, sigma=sigma, tau_g=tau_g)

    def spy_gn(ev, match, p0, **kw):
        seen['admissible'] = kw.get('admissible')
        raise _Stop
    monkeypatch.setattr(douse.SpectrumEvaluator, 'set_target', spy_target)
    monkeypatch.setattr('douse.optimize.gauss_newton', spy_gn)
    spec = dataclasses.replace(im.load(SPECS/'inverse_pilot_v1.toml'), box=True)
    all_runs = im.runs(spec)
    im.run_one(spec, 0, *all_runs[0])
    assert 'sigma' in seen and callable(seen['admissible'])
