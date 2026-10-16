"""`summarize_grid`'s basins: certified endpoints chained by `F`, and ledgers pooled from a
directory tree of per-task files (`benchmarks/slurm.py`'s layout)."""
import json
import subprocess
import sys

import pytest

sg = pytest.importorskip('benchmarks.summarize_grid')


def row(start, fun, N=6, k=1, success=True, shape=True):
    """A certified-looking schema-1 row. The shape is a stand-in, distinct per basin (the
    fingerprint merges basins whose shapes agree), so the two values stay two basins;
    `shape=False` leaves it out and the basins are keyed on `F` alone."""
    bend = (round(fun, 3) - 3.0)*1000
    angles = sorted(180.0*(N - 2)/N + (bend if j % 2 else -bend) for j in range(N))
    r = dict(N=N, k=k, start=start, fun=fun, worst_sigma=1e-10, worst_weyl_deficit=0.0,
             nit=10, seconds=1.0, success=success, resolution=1e-6, message='ok', ok=True)
    return {**r, 'edges': [4.0/N]*N, 'angles': angles} if shape else r


# The overnight N=6 k=2 (0-based k=1) picture: two genuine local maxima.
TWO_BASINS = [row(0, 3.1922), row(1, 3.1981), row(2, 3.19220000001), row(3, 3.1981000002),
              row(4, 3.1922000003), row(5, 3.1922000001), row(6, 3.1981000001),
              row(7, 3.30, success=False)]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_two_basins_are_counted_with_their_best_member():
    s = sg.summarize(TWO_BASINS, sigma_max=1e-6, res_max=1e-4)
    bs = s['basins']
    assert [len(b['rows']) for b in bs] == [3, 4]
    assert bs[0]['fun'] == pytest.approx(3.1981000002) and bs[0]['best']['start'] == 3
    assert bs[1]['fun'] == pytest.approx(3.1922000003) and bs[1]['best']['start'] == 4
    assert s['n_certified'] == 7 and s['n_agree'] == 3     # the uncertified 3.30 is not a basin
    assert sg.basin_line(bs).startswith('3.1981000002 x3 (s3) | 3.1922000003 x4 (s4)')


def test_basins_chain_neighbours_and_the_tolerance_is_relative():
    # 1e-6 relative steps: each neighbour within tolerance, the ends 2.0e-6 apart -> ONE basin.
    rows = [row(i, 3.0*(1 + i*0.9e-6), shape=False) for i in range(3)]
    assert len(sg.summarize(rows, 1e-6, 1e-4)['basins']) == 1
    assert len(sg.summarize(rows, 1e-6, 1e-4, basin_rtol=1e-7)['basins']) == 3
    assert len(sg.summarize([{**r, 'angles': None} for r in TWO_BASINS], 1e-6, 1e-4,
                            basin_rtol=1e-2)['basins']) == 1


def test_a_directory_tree_of_task_ledgers_pools_per_cell(tmp_path):
    for i, r in enumerate(TWO_BASINS):
        write(tmp_path / 'N6_k1' / f'task_{i}.jsonl', [r])
    write(tmp_path / 'N4_k1.jsonl', [row(0, 2.5, N=4), row(1, 2.5, N=4)])
    write(tmp_path / 'archive' / 'N6_k1.jsonl', [row(0, 9.9)])       # not a cell directory
    found = sg.cell_ledgers(str(tmp_path))
    assert sorted(found) == ['N4_k1', 'N6_k1'] and len(found['N6_k1']) == 8
    rows = sg.load_cell(found['N6_k1'])
    assert sorted(r['start'] for r in rows) == list(range(8))
    assert len(sg.summarize(rows, 1e-6, 1e-4)['basins']) == 2


def test_a_retried_task_counts_once(tmp_path):
    """Task 3 resubmitted: its ledger holds a crash and then the result, and a stray copy of the
    same start in another file must not add a ninth start to the cell."""
    write(tmp_path / 'N6_k1' / 'task_3.jsonl',
          [dict(N=6, k=1, start=3, ok=False, error='boom'), TWO_BASINS[3]])
    write(tmp_path / 'N6_k1' / 'task_3b.jsonl', [TWO_BASINS[3]])
    rows = sg.load_cell(sg.cell_ledgers(str(tmp_path))['N6_k1'])
    assert len(rows) == 1 and rows[0]['ok']


def test_the_cli_prints_the_basins_from_a_tree(tmp_path):
    for i, r in enumerate(TWO_BASINS):
        write(tmp_path / 'N6_k1' / f'task_{i}.jsonl', [r])
    out = subprocess.run([sys.executable, '-m', 'benchmarks.summarize_grid', '--dir',
                          str(tmp_path), '--basin-rtol', '1e-6'],
                         capture_output=True, text=True, check=True).stdout
    assert '1 cell(s)' in out
    assert '3.1981000002 x3 (s3) | 3.1922000003 x4 (s4)' in out
    assert 'cells with more than one certified optimum' in out
