"""`benchmarks/family_grid.py`: the grid is a function of the spec, tasks partition it, and one
solve is checked against a closed-form spectrum."""
import json
import pathlib

import numpy as np
import pytest

fg = pytest.importorskip('benchmarks.family_grid')
SPECS = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs'
TRI = SPECS / 'triangle_grid_v0.toml'


def test_grid_is_in_the_fundamental_domain_with_the_angle_floor():
    spec = fg.load(TRI)
    pts = fg.points(spec)
    assert len(pts) == 2462 and pts == fg.points(spec)
    for a, b in pts:
        assert fg._tri_ok(a, b)
        v = fg._tri(a, b)
        ang = fg.interior_angles_deg(v)
        assert ang.min() >= spec.min_angle_deg - 1e-9 and abs(ang.sum() - 180) < 1e-9
        assert fg.area(v) > 0


@pytest.mark.parametrize('family', sorted(fg.FAMILIES))
def test_every_family_is_counter_clockwise_and_convex(family):
    fam = fg.FAMILIES[family]
    spec = fg.GridSpec(name='t', family=family, h=0.1)
    pts = fg.points(spec)
    assert pts
    for a, b in pts:
        v = fam['vertices'](a, b)
        assert fg.area(v) > 0
        assert np.all(fg.interior_angles_deg(v) < 180 + 1e-9)


def test_tasks_partition_the_grid_and_scheduling_is_not_content(tmp_path):
    spec = fg.load(TRI)
    n = len(fg.points(spec))
    T = spec.tasks
    ids = sorted(i for t in range(T) for i in range(t, n, T))
    assert ids == list(range(n))
    src = TRI.read_text().replace('tasks = 200', 'tasks = 7').replace('"8:00:00"', '"1:00:00"')
    (tmp_path / 'r.toml').write_text(src)
    assert fg.load(tmp_path / 'r.toml').sha == spec.sha
    (tmp_path / 'typo.toml').write_text(src + 'n_eig = 3\n')
    with pytest.raises(ValueError, match='unknown key'):
        fg.load(tmp_path / 'typo.toml')


def test_coarse_bracket_is_content_only_when_set(tmp_path):
    v0 = fg.load(TRI)
    assert not v0.coarse_bracket and 'coarse_bracket' not in v0.content()
    src = TRI.read_text()
    (tmp_path / 'off.toml').write_text(src + 'coarse_bracket = false\n')
    (tmp_path / 'on.toml').write_text(src + 'coarse_bracket = true\n')
    assert fg.load(tmp_path / 'off.toml').sha == v0.sha
    assert fg.load(tmp_path / 'on.toml').sha != v0.sha
    v1 = fg.load(TRI.with_name('triangle_grid_v1.toml'))
    assert v1.coarse_bracket and fg.points(v1) == fg.points(v0)


def test_sbatch_is_one_array_over_the_tasks():
    spec = fg.load(TRI)
    text = fg.sbatch_text(spec, 'benchmarks/specs/triangle_grid_v0.toml', 'out', '4G',
                          '12:00:00', 'normal_q', 'acct', 'uv run python')
    assert '#SBATCH --array=0-199' in text and '#SBATCH --account=acct' in text
    assert '--task ${SLURM_ARRAY_TASK_ID} --ntasks 200' in text
    assert 'export OPENBLAS_NUM_THREADS=1' in text


@pytest.mark.slow
def test_right_isosceles_against_the_half_square(tmp_path):
    """Apex (1/2, 1/2): half of the square of side 1/sqrt(2), whose Dirichlet spectrum is
    2 pi^2 (m^2 + n^2), m > n >= 1. Also exercises the ledger and resume."""
    toml = tmp_path / 's.toml'
    toml.write_text('name = "s"\nfamily = "triangle"\nh = 0.5\nn_eigs = 6\nmin_angle_deg = 10\n')
    spec = fg.load(toml)
    pts = fg.points(spec)
    i = pts.index((0.5, 0.5))
    row = fg.solve_point(spec, i, 0.5, 0.5)
    exact = sorted(2*np.pi**2*(m*m + n*n) for m in range(1, 12) for n in range(1, m))[:6]
    assert row['status'] == 'ok'
    assert np.allclose(row['unfolded'], exact, rtol=1e-8)
    assert row['weyl_deficit'] < 0.5
    json.dumps(row)


# ---- suspect rows and repair ----------------------------------------------------------------------

def _row(i, lam, mult=None, bound=None, flags=(), status='ok', sha=None, **kw):
    mult = mult or [1]*len(lam)
    bound = bound or [1e-9]*len(lam)
    unf = [x for x, m in zip(lam, mult) for _ in range(m)]
    return dict(id=i, a=0.3, b=0.3, status=status, lam=lam, mult=mult, bound=bound,
                met=[True]*len(lam), unfolded=unf, flags=list(flags), area=0.15, perim=2.0,
                wall=1.0, spec_sha=sha, **kw)


def test_two_simple_levels_inside_their_bounds_are_one_eigenvalue_found_twice():
    """The pilot's (0.15, 0.20): 1729.019 twice, 1e-10 apart, bounds 1e-5 -- a duplicate, which
    shifts every level above it. A real double is ONE level of multiplicity 2 and is not
    suspect; nor are well-separated levels with loose bounds."""
    dup = _row(0, [10.0, 20.0, 20.0 + 1e-7, 30.0], bound=[1e-9, 1e-5, 1e-5, 1e-9])
    assert fg.suspect(dup) == ['overlapping levels 2,3']
    double = _row(0, [10.0, 20.0, 30.0], mult=[1, 2, 1], bound=[1e-9, 1e-5, 1e-9])
    assert fg.suspect(double) == []
    loose = _row(0, [10.0, 20.0, 30.0], bound=[1e-3, 1e-3, 1e-3])
    assert fg.suspect(loose) == []


def test_the_count_audit_and_a_bad_status_make_a_row_suspect_and_the_weyl_deficit_does_not():
    assert fg.suspect(_row(0, [1.0, 2.0], flags=['count audit: excess (mean Weyl deficit -0.7)'])) \
        == ['count audit']
    assert fg.suspect(_row(0, [1.0, 2.0], status='short')) == ['status short']
    assert fg.suspect(_row(0, [1.0, 2.0], weyl_deficit=0.9)) == []


def test_a_level_straddling_n_eigs_is_kept_whole():
    assert fg._keep_levels([1, 1, 2, 1], 3) == 3       # 1 + 1 + 2 covers 3; the double stays
    assert fg._keep_levels([1, 1, 1, 1], 2) == 2
    assert fg._keep_levels([1, 1], 5) == 2


def test_collect_prefers_a_repaired_row_and_leaves_out_one_still_suspect(tmp_path):
    spec = fg.load(SPECS / 'triangle_pilot_v1.toml')
    dup = dict(bound=[1e-9, 1e-5, 1e-5, 1e-9])
    with open(tmp_path / 'task_0.jsonl', 'w') as fh:
        fh.write(json.dumps(_row(0, [10.0, 20.0, 20.0 + 1e-7, 30.0], sha=spec.sha, **dup)) + '\n')
        fh.write(json.dumps(_row(1, [10.0, 20.0, 20.0 + 1e-7, 30.0], sha=spec.sha, **dup)) + '\n')
        fh.write(json.dumps(_row(2, [11.0, 21.0, 31.0, 41.0], sha=spec.sha)) + '\n')
    (tmp_path / 'repair').mkdir()
    with open(tmp_path / 'repair' / 'task_0.jsonl', 'w') as fh:
        fh.write(json.dumps(_row(0, [10.0, 20.0, 30.0, 40.0], sha=spec.sha)) + '\n')   # fixed
    rows, rep = fg.collect(spec, tmp_path)
    assert rows[0].get('repaired') and rows[0]['lam'] == [10.0, 20.0, 30.0, 40.0]
    assert 'suspect rows (count audit, duplicated or unresolved levels, last-rung fallback): 2; ' \
           'repaired 1, awaiting repair 1' in rep
    _, _, _, _, ids = fg.to_arrays(spec, rows)
    assert list(ids) == [0, 2]                                # id 1 is still suspect: left out


def test_the_repair_job_is_its_own_array_over_its_own_ledgers():
    spec = fg.load(SPECS / 'triangle_pilot_v1.toml')
    text = fg.sbatch_text(spec, 'benchmarks/specs/triangle_pilot_v1.toml', 'out', '1G', '1:00:00',
                          None, None, '.venv/bin/python', cmd='repair', ntasks=4)
    assert '#SBATCH --array=0-3' in text and '--output=out/repair/logs/' in text
    assert 'family_grid repair benchmarks/specs/triangle_pilot_v1.toml' in text and '--ntasks 4' in text


def test_a_level_whose_bound_is_as_large_as_itself_is_not_an_eigenvalue():
    """Full triangle grid: phantom levels with bounds 0.3..18 x their value, three of them
    invisible to the count audit. Real levels -- even the flat strip's, short of prec -- stay
    far below the cut."""
    phantom = _row(0, [10.0, 20.0, 30.0], bound=[1e-9, 1e-9, 9.9])          # the top level
    assert fg.suspect(phantom) == ['unresolved level 3']
    short_of_prec = _row(0, [10.0, 20.0, 30.0], bound=[3e-7, 1e-6, 3e-6])   # rel <= 1e-7
    assert fg.suspect(short_of_prec) == []


def test_slurm_mail_is_one_notice_per_array_and_comes_from_the_environment(monkeypatch):
    """END,FAIL -- never ARRAY_TASKS, which mails every task. The address is never in a spec
    (public repository): `--mail`, else `$DOUSE_SLURM_MAIL`, else no mail lines at all."""
    from benchmarks.slurm import mail_address, mail_lines
    spec = fg.load(SPECS / 'triangle_pilot_v1.toml')
    monkeypatch.delenv('DOUSE_SLURM_MAIL', raising=False)
    assert mail_address() is None and mail_lines(None) == []
    plain = fg.sbatch_text(spec, 's.toml', 'out', '1G', '1:00:00', None, None, 'py')
    assert '--mail' not in plain
    monkeypatch.setenv('DOUSE_SLURM_MAIL', 'someone@example.edu')
    text = fg.sbatch_text(spec, 's.toml', 'out', '1G', '1:00:00', None, None, 'py',
                          mail=mail_address())
    assert '#SBATCH --mail-type=END,FAIL' in text and '#SBATCH --mail-user=someone@example.edu' in text
    assert 'ARRAY_TASKS' not in text
    assert mail_address('other@example.edu') == 'other@example.edu'


def test_a_fallback_to_the_planners_last_rung_is_suspect_and_an_earlier_one_is_not():
    """Full triangle grid: 61% of last-rung fallbacks were broken, 0.6% of earlier ones."""
    last = 'planner: no rung met prec=1e-10; returning rung 4 of 4, the one with the most levels met'
    early = 'planner: no rung met prec=1e-10; returning rung 2 of 4, the one with the most levels met'
    assert fg.suspect(_row(0, [1.0, 2.0], flags=[last])) == ['planner fell back to its last rung']
    assert fg.suspect(_row(0, [1.0, 2.0], flags=[early])) == []
    repaired = _row(0, [1.0, 2.0], flags=[last], repair_of=['planner fell back to its last rung'])
    assert fg.suspect(repaired) == []                    # a clean repair is kept, whatever its rung
