"""The ledger: one JSON row per run, with a schema.  (2026-10-14)

191 rows were on disk in four different key sets, none written by the code that read them, and
the readers used lappy's names (`worst_sigma`, a tension) for what chladni writes (a certified
bound). `SCHEMA = 2` rows carry chladni's names (`douse.backend.RunHealth.as_row`) plus
`schema`; `normalize` maps a row of any era onto BOTH name sets, so a reader may use the v2
names and an old synthetic row still reads. `read_rows` dedupes by start, last attempt wins,
which is the resume semantics (`run_grid` retries a crashed start and the runner appends).
"""
import json
import os

import numpy as np

__all__ = ['SCHEMA', 'V1_TO_V2', 'normalize', 'read_rows', 'write_row', 'result_row',
           'schema_of', 'schema_mix']

SCHEMA = 2

#: v1 name -> v2 name. The comparison ledger (`chladni_vs_lappy`) used `seed`/`wall` where the
#: multistart ledger used `start`/`seconds`.
V1_TO_V2 = {
    'worst_sigma': 'worst_bound',
    'worst_weyl_deficit': 'worst_deficit',
    'n_weyl_flags': 'n_audit_flags',
    'n_weyl_refusals': 'n_audit_refusals',
    'n_spurious_refusals': 'n_unmet_refusals',
    'plan_at': 'plan_built_at',
    'n_tracks': 'n_warm',
    'n_track_refusals': 'n_warm_refused',
    'seed': 'start',
    'wall': 'seconds',
}


def schema_of(row):
    return int(row.get('schema', 1) or 1)


def normalize(row):
    """A copy of `row` carrying both the v1 and the v2 names for every statistic it has, and
    `schema`. Values are not changed: a v1 `worst_sigma` (a tension) becomes `worst_bound` by
    NAME only, and `schema=1` says so."""
    out = dict(row)
    out['schema'] = schema_of(row)
    for old, new in V1_TO_V2.items():
        if old in out and new not in out:
            out[new] = out[old]
        elif new in out and old not in out:
            out[old] = out[new]
    return out


def read_rows(path, dedupe=True):
    """The rows of one ledger, normalized; deduped by start (last attempt wins) unless asked
    not to. Rows with no `start` are kept as they are."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(normalize(json.loads(line)))
    if not dedupe:
        return rows
    last = {}
    for i, r in enumerate(rows):
        if r.get('start') is not None:
            last[r['start']] = i
    keep = set(last.values())
    return [r for i, r in enumerate(rows) if r.get('start') is None or i in keep]


def write_row(path, row):
    """Append one row, flushed to disk, so a crash loses at most the row being written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as fh:
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def _clean(v):
    if isinstance(v, np.ndarray):
        return [_clean(t) for t in v.tolist()]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_clean(t) for t in v]
    if isinstance(v, dict):
        return {k: _clean(t) for k, t in v.items()}
    return v


def result_row(res, health=None, max_trace=400, **meta):
    """A schema-2 row from a `maximize_ratio` result and the backend's `RunHealth` (or the
    result's own `health` dict). `meta` is the run's identity and provenance (N, k, start,
    backend, prec, SHAs, ...). The v2 names only; `normalize` gives a reader the v1 ones."""
    h = health.as_row() if health is not None and hasattr(health, 'as_row') else \
        (health if health is not None else res.get('health'))
    row = dict(schema=SCHEMA, **meta)
    row.update(fun=float(res.fun), nit=int(res.nit), x=_clean(res.x), lams=_clean(res.lams),
               gap=_clean(res.get('gap', np.nan)), delta=_clean(res.get('delta', np.nan)),
               resolution=_clean(res.get('resolution', np.nan)),
               success=bool(res.success), message=str(res.message),
               verified=bool(res.get('verified', True)),
               n_warnings=int(res.get('n_warnings', 0)),
               n_cluster_steps=int(res.get('n_cluster_steps', 0)),
               n_interior_steps=int(res.get('n_interior_steps', 0)),
               n_qn_steps=int(res.get('n_qn_steps', 0)),
               n_degeneracy_moves=int(res.get('n_degeneracy_moves', 0)),
               n_discontinuity_repairs=int(res.get('n_discontinuity_repairs', 0)),
               n_infeasible=int(res.get('n_infeasible', 0)),
               n_unresolved=int(res.get('n_unresolved', 0)),
               n_solver_refused=int(res.get('n_solver_refused', 0)))
    if h is not None:
        h = dict(h)
        trace = h.pop('trace', None)
        row.update(_clean(h))
        if trace is not None:
            row['trace'] = _clean(list(trace)[:max_trace])
    return row


def schema_mix(rows):
    """`{schema: count}` over the rows, for a summary to print beside its counts."""
    mix = {}
    for r in rows:
        mix[schema_of(r)] = mix.get(schema_of(r), 0) + 1
    return mix
