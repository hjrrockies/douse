"""`benchmarks/slurm.py`: one array job per cell, one start per task, written and never submitted."""
import pathlib
import re

import pytest

from douse import spec as S

sl = pytest.importorskip('benchmarks.slurm')
SPECS = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs'
V0 = SPECS / 'thesis_ratio_v0.toml'


def test_slurm_fields_are_scheduling_not_content(tmp_path):
    """Adding resources to a spec must not change the experiment's identity."""
    v0 = S.load(V0)
    assert v0.sha == '8476c13e692e'             # the hash every v0 row already cites
    assert v0.slurm_mem is None and v0.slurm_time is None
    src = V0.read_text()
    (tmp_path / 'r.toml').write_text(src + 'slurm_mem = "8G"\nslurm_time = "48:00:00"\n'
                                     'slurm_partition = "normal_q"\nslurm_account = "douse"\n')
    r = S.load(tmp_path / 'r.toml')
    assert r.sha == v0.sha and r.slurm_mem == '8G' and r.slurm_partition == 'normal_q'
    assert not set(S.SCHEDULING) & set(r.content())
    (tmp_path / 'typo.toml').write_text(src + 'slurm_memory = "8G"\n')
    with pytest.raises(ValueError, match='unknown key'):
        S.load(tmp_path / 'typo.toml')
    (tmp_path / 'num.toml').write_text(src + 'slurm_mem = 8\n')
    with pytest.raises(ValueError, match='slurm_mem'):
        S.load(tmp_path / 'num.toml')


def test_the_dry_run_on_v0_prints_one_array_job_per_cell(capsys, monkeypatch):
    def no_sbatch(*a, **kw):
        raise AssertionError('slurm.py must never run a command')
    monkeypatch.setattr('subprocess.run', no_sbatch)
    monkeypatch.setattr('subprocess.Popen', no_sbatch)
    sl.main([str(V0), '--dry-run'])
    out = capsys.readouterr().out
    v0 = S.load(V0)
    assert out.count('#!/bin/bash') == len(v0.cells) == 12
    assert out.count('#SBATCH --array=0-7') == 12
    assert out.count('#SBATCH --mem=4G') == 12 and '#SBATCH --time=24:00:00' in out
    assert '--partition' not in out and '--account' not in out
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_NUM_THREADS'):
        assert out.count(f'export {var}=1') == 12
    cmd = [ln for ln in out.splitlines() if 'benchmarks.ratio_multistart' in ln]
    assert len(cmd) == 12
    assert ('--spec benchmarks/specs/thesis_ratio_v0.toml --N 6 --k 2 --start-index '
            '${SLURM_ARRAY_TASK_ID} --out "benchmarks/run/slurm/thesis_ratio_v0/N6_k2/'
            'task_${SLURM_ARRAY_TASK_ID}.jsonl"') in out
    assert re.findall(r'^sbatch (\S+)$', out, re.M)[0].endswith('thesis_ratio_v0/N3_k1.sbatch')
    assert v0.sha in out


def test_spec_resources_are_used_and_flags_win(tmp_path):
    (tmp_path / 'r.toml').write_text(V0.read_text() + 'slurm_mem = "8G"\nslurm_time = "2:00:00"\n'
                                     'slurm_partition = "normal_q"\n')
    spec = S.load(tmp_path / 'r.toml')
    res = sl.resources(spec)
    assert res == dict(mem='8G', time='2:00:00', partition='normal_q', account=None)
    assert sl.resources(spec, mem='16G')['mem'] == '16G'
    text = sl.script(spec, 6, 2, 'r.toml', 'out', res)
    assert '#SBATCH --mem=8G' in text and '#SBATCH --partition=normal_q' in text
    assert '--account' not in text


def test_writing_creates_the_scripts_and_the_log_directory(tmp_path):
    sl.main([str(V0), '--out', str(tmp_path / 'jobs')])
    scripts = sorted((tmp_path / 'jobs').glob('*.sbatch'))
    assert len(scripts) == 12 and (tmp_path / 'jobs' / 'logs').is_dir()
    assert 'N6_k2_%A_%a.out' in (tmp_path / 'jobs' / 'N6_k2.sbatch').read_text()


def test_ratio_multistart_takes_one_start_of_one_spec_cell(tmp_path, monkeypatch):
    """The CLI the array task runs: `--start-index` restricts to one start, `--out FILE` is the
    task ledger, `--N/--k` must name a cell of the spec. No optimizer is run here."""
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    seen = []

    def fake_run_one(N, k, i, *a, **kw):
        seen.append((N, k, i))
        return dict(N=N, k=k, start=i, ok=False, error='fake')
    monkeypatch.setattr(rm, 'run_one', fake_run_one)
    monkeypatch.setattr(rm, 'partition', lambda rows, s: ([], [], [], [], []))
    ledger = tmp_path / 'N6_k2' / 'task_5.jsonl'
    monkeypatch.setattr('sys.argv', ['x', '--spec', str(V0), '--N', '6', '--k', '2',
                                     '--start-index', '5', '--out', str(ledger)])
    rm.main()
    assert seen == [(6, 2, 5)] and ledger.exists()
    monkeypatch.setattr('sys.argv', ['x', '--spec', str(V0), '--N', '9', '--k', '2'])
    with pytest.raises(SystemExit):
        rm.main()
    monkeypatch.setattr('sys.argv', ['x', '--spec', str(V0), '--N', '6', '--k', '2',
                                     '--start-index', '8'])
    with pytest.raises(SystemExit):
        rm.main()
