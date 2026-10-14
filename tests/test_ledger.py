"""`douse.ledger`: rows of any era read under one set of names; new rows carry the schema."""
import json

import numpy as np

from douse import ledger
from douse.backend import RunHealth, SolveTrace
from douse.optimize import OptimizeResult


def test_a_v1_row_reads_under_both_names_and_says_it_is_v1():
    r = ledger.normalize(dict(start=3, fun=2.5, worst_sigma=1e-9, worst_weyl_deficit=0.1,
                              n_weyl_flags=2, plan_at=[0.1, 0.2]))
    assert r['schema'] == 1
    assert r['worst_bound'] == 1e-9 and r['worst_sigma'] == 1e-9
    assert r['worst_deficit'] == 0.1 and r['n_audit_flags'] == 2
    assert r['plan_built_at'] == [0.1, 0.2]


def test_a_comparison_ledger_row_maps_seed_and_wall():
    r = ledger.normalize(dict(backend='chladni', seed=0, wall=12.0, fun=2.5))
    assert r['start'] == 0 and r['seconds'] == 12.0


def test_a_v2_row_also_carries_the_v1_names_for_old_readers():
    r = ledger.normalize(dict(schema=2, start=0, worst_bound=1e-10, worst_deficit=0.05))
    assert r['schema'] == 2 and r['worst_sigma'] == 1e-10 and r['worst_weyl_deficit'] == 0.05


def test_read_rows_dedupes_by_start_keeping_the_last(tmp_path):
    path = tmp_path / 'N4_k1.jsonl'
    for row in (dict(start=0, fun=1.0), dict(start=1, fun=2.0, ok=False, error='x'),
                dict(start=1, fun=2.5), dict(fun=9.0)):
        ledger.write_row(path, row)
    rows = ledger.read_rows(path)
    assert [r.get('start') for r in rows] == [0, 1, None]
    assert rows[1]['fun'] == 2.5 and all(r['schema'] == 1 for r in rows)
    assert len(ledger.read_rows(path, dedupe=False)) == 4
    assert ledger.read_rows(tmp_path / 'missing.jsonl') == []


def test_result_row_is_schema_2_with_the_health_and_a_capped_trace():
    res = OptimizeResult(x=np.array([0.1, 0.2]), fun=2.5, nit=7, lams=np.array([1.0, 2.5]),
                         gap=1.5, delta=0.01, resolution=np.nan, success=True, message='ok',
                         n_warnings=1, n_cluster_steps=3)
    h = RunHealth(worst_bound=1e-10, worst_deficit=0.1, n_replans=2,
                  trace=tuple(SolveTrace('cold', None, 1, 100, 1.0) for _ in range(5)))
    row = ledger.result_row(res, health=h, N=4, k=1, start=0, backend='chladni', max_trace=3)
    assert row['schema'] == 2 and row['N'] == 4 and row['backend'] == 'chladni'
    assert row['worst_bound'] == 1e-10 and row['n_replans'] == 2 and row['resolution'] is None
    assert row['x'] == [0.1, 0.2] and row['n_cluster_steps'] == 3 and row['n_warnings'] == 1
    assert len(row['trace']) == 3 and row['trace'][0]['kind'] == 'cold'
    json.dumps(row)                                              # JSON-clean
    assert ledger.schema_mix([row, dict(fun=1.0)]) == {2: 1, 1: 1}
