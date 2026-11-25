"""`benchmarks/resume.py`: which tasks of a spec's Slurm runs to resubmit, and at what memory."""
import json
import pathlib

import pytest

from douse import spec as S

rs = pytest.importorskip('benchmarks.resume')
V0 = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs' / 'thesis_ratio_v0.toml'
OOM_LOG = ('[2026-11-21T17:28:05.088] error: Detected 1 oom_kill event in StepId=8712284.batch. '
           'Some of the step tasks have been OOM Killed.\n')


@pytest.fixture
def spec(tmp_path):
    (tmp_path / 's.toml').write_text(
        V0.read_text() + 'slurm_mem = {"3-5" = "2G", "6" = "3G"}\n')
    return S.load(tmp_path / 's.toml')


def _row(spec, start, ok=True):
    return json.dumps(dict(start=start, ok=ok, spec_sha=spec.sha, F=1.0)) + '\n'


def test_sizes_round_trip():
    assert rs.gib('2G') == 2 and rs.gib('512M') == 0.5 and rs.gib('1T') == 1024
    assert rs.as_mem(4.0) == '4G' and rs.as_mem(1.5) == '1536M'
    with pytest.raises(ValueError):
        rs.gib('lots')
    assert rs.array_spec([0, 1, 2, 5, 7, 8]) == '0-2,5,7-8' and rs.array_spec([3]) == '3'


def test_unfinished_tasks_come_back_at_their_cells_memory_and_ooms_go_up_a_tier(spec, tmp_path):
    run = tmp_path / 'runs'
    (run / 'logs').mkdir(parents=True)
    cell = run / 'N3_k1'
    cell.mkdir()
    # starts 0 and 1 finished; start 2 crashed (retried); start 3 finished under ANOTHER spec
    (cell / 'task_0.jsonl').write_text(_row(spec, 0))
    (cell / 'task_1.jsonl').write_text(_row(spec, 1))
    (cell / 'task_2.jsonl').write_text(_row(spec, 2, ok=False))
    (cell / 'task_3.jsonl').write_text(json.dumps(dict(start=3, ok=True, spec_sha='other')) + '\n')
    # start 4 was OOM-killed once, start 5 twice (two jobs); start 6 timed out (no OOM line)
    (run / 'logs' / 'N3_k1_100_4.out').write_text('...\n' + OOM_LOG)
    (run / 'logs' / 'N3_k1_100_5.out').write_text(OOM_LOG)
    (run / 'logs' / 'N3_k1_200_5.out').write_text(OOM_LOG)
    (run / 'logs' / 'N3_k1_100_6.out').write_text('CANCELLED DUE TO TIME LIMIT\n')
    (run / 'logs' / 'N3_k1_100_14.out').write_text(OOM_LOG)    # task 14 is not task 4
    live = {('N3_k1', 7)}                                         # still running: left alone
    plan = [(c, m, t) for c, m, t in rs.plan(spec, str(run), live) if c == 'N3_k1']
    assert plan == [('N3_k1', '2G', [2, 3, 6]), ('N3_k1', '4G', [4]), ('N3_k1', '8G', [5])]
    # a cell with no runs at all comes back whole, at its own N's figure
    n6 = [(m, t) for c, m, t in rs.plan(spec, str(run)) if c == 'N6_k1']
    assert n6 == [('3G', list(range(8)))]


def test_the_cap_and_a_flag_bound_the_memory(spec, tmp_path):
    run = tmp_path / 'runs'
    (run / 'logs').mkdir(parents=True)
    for job in range(5):
        (run / 'logs' / f'N3_k1_{job}_0.out').write_text(OOM_LOG)
    first = lambda **kw: next((m, t) for c, m, t in rs.plan(spec, str(run), **kw)
                              if c == 'N3_k1' and 0 in t)
    assert first() == ('16G', [0])                       # 2G x 2^5, capped at 16G
    assert first(cap='8G') == ('8G', [0])
    assert first(mem='1G', cap='64G') == ('32G', [0])     # a flag replaces the spec's figure


def test_main_prints_sbatch_lines_and_runs_nothing(spec, tmp_path, capsys, monkeypatch):
    def no_run(*a, **kw):
        raise AssertionError('resume.py must never run a command with --no-squeue')
    monkeypatch.setattr('subprocess.run', no_run)
    run = tmp_path / 'runs'
    rs.main([spec.source, '--dir', str(run), '--no-squeue', '--qos', 'tc_normal_long'])
    out = capsys.readouterr().out
    assert f'sbatch --array=0-7 --mem=2G --qos=tc_normal_long {run}/N3_k1.sbatch' in out
    assert f'sbatch --array=0-7 --mem=3G --qos=tc_normal_long {run}/N6_k3.sbatch' in out
    assert '96 unfinished task(s) in 12 cell(s)' in out


def test_a_table_flag_sets_each_cells_base(spec, tmp_path):
    run = tmp_path / 'runs'
    got = {c: m for c, m, _ in rs.plan(spec, str(run), mem='3-5=2G,6=3G')}
    assert got['N3_k1'] == '2G' and got['N6_k1'] == '3G'


def test_a_refusal_stop_is_reported_not_resubmitted(spec, tmp_path, capsys):
    run = tmp_path / 'runs'
    cell = run / 'N3_k1'
    cell.mkdir(parents=True)
    bad = lambda i, err: json.dumps(dict(start=i, ok=False, spec_sha=spec.sha, error=err)) + '\n'
    refusal = 'UnresolvedSpectrum: chladni: planner: no rung met prec=1e-09'
    (cell / 'task_0.jsonl').write_text(bad(0, refusal))                     # refused: reported
    (cell / 'task_1.jsonl').write_text(bad(1, 'SolverRefusal: no spectrum') + bad(1, refusal))
    (cell / 'task_2.jsonl').write_text(bad(2, refusal) + bad(2, 'MemoryError: '))   # last is a crash
    (cell / 'task_3.jsonl').write_text(bad(3, 'ValueError: a bug'))         # a crash: retried
    stops = []
    tasks = [t for c, _, t in rs.plan(spec, str(run), stops=stops) if c == 'N3_k1'][0]
    assert tasks == [2, 3, 4, 5, 6, 7]
    assert [(c, i) for c, i, _ in stops] == [('N3_k1', 0), ('N3_k1', 1)]
    forced = [t for c, _, t in rs.plan(spec, str(run), retry_refused=True) if c == 'N3_k1'][0]
    assert forced == list(range(8))
    rs.main([spec.source, '--dir', str(run), '--no-squeue'])
    out = capsys.readouterr().out
    assert '# 2 refusal stop(s) NOT resubmitted' in out and '#   N3_k1 s0: UnresolvedSpectrum' in out
    assert 'sbatch --array=2-7 --mem=2G' in out
