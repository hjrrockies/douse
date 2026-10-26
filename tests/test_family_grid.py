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
