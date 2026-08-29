"""The console summary's gates, which are what a person actually watches a sweep produce.

`summarize_grid` is the tool of record and always filtered correctly. This file covers the OTHER
reporting path: `benchmarks/ratio_multistart.py`'s own summary, which applied neither gate and
so printed the impossible 2.6203171651 as `group 0` -- the leading line, above the true answer,
with `0 converged` as the only hint.
"""
import json

import numpy as np
import pytest

from douse import spectral_guard

rm = pytest.importorskip('benchmarks.ratio_multistart')


def row(start, fun, sigma=1e-10, deficit=0.0, ok=True):
    return dict(N=6, k=1, start=start, fun=fun, worst_sigma=sigma, worst_weyl_deficit=deficit,
                nit=10, seconds=1.0, success=True, resolution=1e-6, message='ok', ok=ok)


def test_a_drifted_basis_is_excluded_from_the_grouping():
    _, drifted, _, good = rm.partition([row(0, 2.53), row(1, 3.63, sigma=6.3e-04)], 1e-6)
    assert [r['start'] for r in drifted] == [1]
    assert [r['start'] for r in good] == [0]


def test_a_missed_mode_is_excluded_THOUGH_ITS_SIGMA_IS_PRISTINE():
    """The 2.6203 failure exactly: sigma = 1.3e-09, clean by every basis measure, mode missing."""
    _, drifted, skipped, good = rm.partition(
        [row(0, 2.53), row(2, 2.6203171651, sigma=1.3e-09, deficit=1.01)], 1e-6)
    assert drifted == []
    assert [r['start'] for r in skipped] == [2]
    assert [r['start'] for r in good] == [0]


def test_a_crashed_row_is_counted_and_not_grouped():
    crashed, _, _, good = rm.partition(
        [row(0, 2.53), dict(N=6, k=1, start=1, ok=False, error='EigensolverFailure')], 1e-6)
    assert [r['start'] for r in crashed] == [1]
    assert [r['start'] for r in good] == [0]


def test_the_two_gates_are_the_ones_summarize_grid_uses():
    """Imported, not restated -- so the two summaries cannot drift apart."""
    from benchmarks import summarize_grid as sg
    assert rm.row_deficit is sg.row_deficit and rm.PPW is sg.PPW


def test_PPW_would_catch_the_2_6203_row_if_the_deficit_gate_ever_missed_it():
    """The backstop, and why it cannot replace the deficit gate: it only speaks at k=1."""
    assert 2.6203171651 > rm.PPW + 1e-9
    assert rm.PPW == pytest.approx(2.5387340, abs=1e-6)


def test_the_live_start_line_flags_a_missed_mode():
    assert 'MISSED MODE' in rm.start_line(row(2, 2.62, deficit=1.01))
    assert 'MISSED MODE' not in rm.start_line(row(0, 2.53))


def _ledger_rows():
    import glob
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, 'benchmarks', 'run', 'ratio_multistart',
                                          'N*_k*.jsonl')))
    return [r for p in paths for r in map(json.loads, filter(str.strip, open(p)))
            if r.get('ok', True) and np.isfinite(r.get('worst_weyl_deficit', np.nan))]


def test_the_recomputed_deficit_is_an_ENDPOINT_statistic_not_the_recorded_worst():
    """The fallback in `row_deficit` is weaker than the field it stands in for, and by a lot.

    `worst_weyl_deficit` is the max over every iterate of a run; the recomputation sees only the
    endpoint `x` and `lams`, so it can only ever be a lower bound on it. Checked on every ledger
    row that carries both.
    """
    rows = _ledger_rows()
    if not rows:
        pytest.skip('no ledger rows with a recorded deficit')
    for r in rows:
        q = dict(r)
        recorded = q.pop('worst_weyl_deficit')
        assert rm.row_deficit(q) <= recorded + 1e-9, r['start']


def test_THE_FALLBACK_CAN_FLIP_THE_GATE_and_here_is_the_case_that_does():
    """N=6 `lam_3/lam_1` start 3: recorded worst 8.76, endpoint 0.19, threshold 0.5.

    So a row written before the guard existed -- 71 of the 80 on disk -- is audited by a
    statistic that would have PASSED this run. Pinned so nobody reads the two as equivalent.
    """
    rows = [r for r in _ledger_rows() if r['N'] == 6 and r['k'] == 2 and r['start'] == 3]
    if not rows:
        pytest.skip('the N6_k2 start 3 row is not on disk')
    r = dict(rows[0])
    recorded = r.pop('worst_weyl_deficit')
    assert recorded > spectral_guard.DEFICIT_TOL
    assert rm.row_deficit(r) < spectral_guard.DEFICIT_TOL


def test_plot_optima_and_summarize_grid_agree_on_the_denominator():
    """They did not: a crashed row was dropped before `plot_optima` counted, so N=6 `lam_2/lam_1`
    read 3/7 there and 3/8 in the tool of record. Both count ATTEMPTED starts now."""
    po = pytest.importorskip('benchmarks.plot_optima')
    sg = pytest.importorskip('benchmarks.summarize_grid')
    import glob
    import os
    cells = po.collect(False)
    if not cells:
        pytest.skip('no ledgers on disk')
    for path in sorted(glob.glob(os.path.join(sg.LEDGERS, 'N*_k*.jsonl'))):
        if path.endswith('_convex.jsonl'):
            continue
        rows = sg.load(path)
        if not rows:
            continue
        key = (rows[0]['N'], rows[0]['k'])
        if key in cells:
            assert cells[key]['n_total'] == sg.summarize(rows, 1e-6, 1e-4)['n_total'], key


def test_the_gates_have_exactly_one_definition_across_all_three_readers():
    po = pytest.importorskip('benchmarks.plot_optima')
    sg = pytest.importorskip('benchmarks.summarize_grid')
    assert po.row_deficit is sg.row_deficit and po.PPW is sg.PPW


def test_a_ledger_row_says_which_guards_produced_it():
    """A cell is not always written by one version of the code -- `run_grid` retries a crashed
    start on resume -- so a row has to carry what watched it."""
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    import inspect
    src = inspect.getsource(rm.run_one)
    for field in ('enforce_weyl=', 'deficit_tol=', 'n_weyl_refusals=', 'refusals='):
        assert field in src, field


def test_a_certificate_is_the_STOP_not_a_threshold_on_the_resolution():
    """Section 9's lesson, one layer up. `gain_tol` was replaced by plateau detection because a
    fixed threshold certified according to where each shape's noise floor happened to fall;
    `res_max` was then applied to the plateau VALUE and did the same thing. Measured: of the 20
    rows it rejected, 17 agreed with their cell's best to 1e-6."""
    sg = pytest.importorskip('benchmarks.summarize_grid')
    row = dict(N=5, k=2, start=0, fun=3.0, worst_sigma=1e-11, worst_weyl_deficit=0.0,
               success=True, resolution=6.4e-4, ok=True, nit=10, seconds=1.0, message='plateaued')
    s = sg.summarize([row], sigma_max=1e-6, res_max=1e-4)
    assert s['best'] is not None, 'a plateau certificate above res_max must still certify'
    assert s['n_certified'] == 1
    assert s['n_sharp'] == 0, 'and must be reported as not sharp'


def test_the_sharp_count_still_distinguishes_quality():
    sg = pytest.importorskip('benchmarks.summarize_grid')
    mk = lambda res, f: dict(N=5, k=2, start=0, fun=f, worst_sigma=1e-11,               # noqa
                             worst_weyl_deficit=0.0, success=True, resolution=res,
                             ok=True, nit=10, seconds=1.0, message='plateaued')
    s = sg.summarize([mk(6.4e-4, 3.0), mk(1e-6, 3.0)], sigma_max=1e-6, res_max=1e-4)
    assert s['n_certified'] == 2 and s['n_sharp'] == 1


def test_the_upper_bound_that_matters_lives_in_the_optimizer():
    """Nothing certifies with an absurd plateau, because `maximize_ratio` only runs the plateau
    test when criticality is already under `plateau_max`. That is the bound, and it sits where
    the measurement is."""
    import inspect
    from douse import cluster_opt
    sig = inspect.signature(cluster_opt.maximize_ratio).parameters
    assert 'plateau_max' in sig and sig['plateau_max'].default == 1e-3


def test_basins_are_keyed_on_F_then_merged_by_SHAPE():
    """Neither key works alone, and both failures were measured on real cells.

    Shape alone split N=6 `lam_2/lam_1` in two on values agreeing to 2.6e-07 (one run stopped
    less sharply and sits 0.28 deg away). `F` alone split N=3 `lam_6/lam_1` in two on the same
    triangle to 0.04 deg. Shape distance separates the classes by two orders of magnitude:
    4.1e-04 for the one pair that is the same optimum, 5.0e-02 to 8.0e-01 for the five distinct.
    """
    sg = pytest.importorskip('benchmarks.summarize_grid')
    def row(f, angles, edges):
        return dict(N=3, k=5, start=0, fun=f, angles=angles, edges=edges, worst_sigma=1e-11,
                    success=True, resolution=1e-6, ok=True, worst_weyl_deficit=0.0)
    # the real N3_k6 pair: same triangle, F differing in the fifth figure
    a = row(5.0212066403, [39.72, 55.33, 84.95], [0.94, 1.28, 1.78])
    b = row(5.0207101960, [39.72, 55.37, 84.91], [0.94, 1.28, 1.78])
    assert len(sg.find_basins([a, b])) == 1, 'the same triangle must not read as two basins'
    # a genuinely different optimum, at a shape distance of 8e-01
    c = row(3.0794061264, [17.73, 81.14, 81.14], [0.61, 1.69, 1.69])
    assert len(sg.find_basins([a, c])) == 2


def test_a_second_basin_is_reported_not_discarded():
    """N=4 `lam_4/lam_1` splits 4-4 between a trapezoid and the sqrt(5) rectangle. Reporting only
    the best threw that away."""
    sg = pytest.importorskip('benchmarks.summarize_grid')
    import glob
    import os
    path = os.path.join(sg.LEDGERS, 'N4_k3.jsonl')
    if not os.path.exists(path):
        pytest.skip('N4_k3 ledger not on disk')
    s = sg.summarize(sg.load(path), 1e-6, 1e-4)
    assert len(s['basins']) == 2
    vals = sorted(b['fun'] for b in s['basins'])
    assert vals[1] == pytest.approx(4.2833985583, abs=1e-9)
    assert vals[0] == pytest.approx(3.5, abs=1e-8)
    assert all(len(b['rows']) == 4 for b in s['basins'])


def test_a_start_that_cannot_be_solved_is_redrawn_deterministically():
    """All nine crashed runs in the grid died at the START shape, before a single iterate -- four
    shapes, five of them the same N=3 triangle. A start is an arbitrary draw, so it is redrawn
    and the discard recorded, which keeps the start distribution statable."""
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    import inspect
    src = inspect.getsource(rm.run_one)
    assert 'max_redraws' in src and 'redraws.append' in src
    assert 'i + 1000*attempt' in src, 'the redraw must be deterministic'
    # the probe is an explicit evaluation, not an inspection of a traceback
    assert 'ev.eigenvalues(p0)' in src and 'traceback' not in src.split('for attempt')[1][:600]


def test_a_retried_start_is_counted_ONCE_and_the_retry_is_the_one_kept(tmp_path):
    """`run_grid` retries a crashed start on resume and `ratio_multistart` APPENDS, so a cell can
    hold more rows than it has starts. `N6_k1` held 9 rows for 8 starts and the table read
    `n_total = 9`, making every "certified of total" denominator wrong for that cell. No value was
    ever affected -- the filters run per row and a duplicate crash row certifies nothing -- but a
    wrong denominator is read as evidence about the basin structure.

    The retry wins: it exists because the earlier attempt failed, and it was written by the newer
    code, whose guards the row itself records.
    """
    from benchmarks import summarize_grid as sg

    path = tmp_path/'N6_k1.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in [
        dict(N=6, k=1, start=0, fun=2.53, ok=True),
        dict(N=6, k=1, start=1, ok=False, error='first attempt'),
        dict(N=6, k=1, start=1, ok=False, error='the retry'),
    ]) + '\n')
    rows = sg.load(str(path))
    assert [r['start'] for r in rows] == [0, 1]
    assert rows[1]['error'] == 'the retry'
