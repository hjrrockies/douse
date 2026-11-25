"""Ledger rows for mirror runs whose optimum is in the run record but never reached the ledger.

    python -m benchmarks.salvage_record <run dir or cell dir> [--out-root DIR] [--dry-run]

WHY. Up to lock-3 (c373c7a), `ratio_multistart.run_mirror` wrote a start's ledger row only after
the antisymmetric-Hessian probe: 1 + (N - 2)(N - 1) cold full-polygon solves with no deadline,
no checkpoint and no record output. Measured on F2 (2026-11-25): 39 h at N=13 and 146 h at N=15.
That left 15 finished optimizations in the probe with no ledger row. Some were killed by the 7-day
limit, and a resubmit re-ran the optimizer from the stale checkpoint, then the probe again.
`run_mirror` now writes the row before the probe; this rebuilds it for runs from before that change.

WHAT A SALVAGED ROW IS. The run record (`DOUSE_RECORD=1`; `douse.record`) ends each finished
optimization with a `result` row, the optimizer's own output (`x`, `fun`, `final_case`,
`convergence`, counters, `worst_sigma`). The run's identity and provenance come from the `head` row before it,
and the vertices from the last `iter` row's `geometry`. The vertices are checked against
`result.x` by rebuilding the mirror chart from them. What is NOT there: the probe (`antisym` says
so), the backend's per-solve health trace, `seconds` and `n_resumes`. `salvaged` names the
source. A ledger keeps the last row per start, so a later real row replaces it.

CHECKED against the 58 F2 starts (2026-11-25) that have both a record and a real row. Every field
both carry agrees, apart from one row each in three fields:
- `x`: the canonical gauge (`param_vector`) takes the longest side, and a mirror polygon's two
  equal sides tie at rounding level. It is the same polygon with a different labelling;
  `vertices` agree to 1e-16.
- `prec`: one evaluator dropped back a precision after its last iterate.
- `plan_at`.

A start that already has a ledger row is left alone. The default writes beside the record (in
place); `--out-root` writes the same relative layout under another directory instead.
"""
import argparse
import datetime
import json
import os
import pathlib
import re

import numpy as np

from douse import degeneracy as dg
from douse import ledger
from douse.polygon import FourierMirrorPolygon, MirrorPolygon, param_vector

PERIM = 4.0                               # benchmarks.ratio_multistart.PERIM
RECORD = re.compile(r'task_(\d+)\.jsonl\.start(\d+)\.record\.jsonl$')


def last_result(rows):
    """`(head, result, last iter)` of the latest optimization that finished, or None."""
    head, it, found = None, None, None
    for r in rows:
        if r.get('kind') == 'head':
            head, it = r, None
        elif r.get('kind') == 'iter':
            it = r
        elif r.get('kind') == 'result':
            found = (head, r, it)
    return found


def salvage_row(rows, record_name):
    """The ledger row for a record's latest finished mirror optimization, or None."""
    got = last_result(rows)
    if got is None:
        return None
    head, res, it = got
    if head is None or not head.get('mirror'):
        raise ValueError(f'{record_name}: not a mirror run; only mirror rows are salvaged')
    if it is None or not (it.get('geometry') or {}).get('v'):
        raise ValueError(f'{record_name}: no iterate geometry before the result')
    xs, ys = it['geometry']['v']
    v = np.asarray(xs) + 1j*np.asarray(ys)
    fourier = head.get('chart') == 'fourier'
    fam, q, _ = (FourierMirrorPolygon.from_vertices(v, PERIM, s=head['s']) if fourier
                 else MirrorPolygon.from_vertices(v, PERIM))
    # The last iterate's polygon IS the result's: its chart coordinates must agree.
    if not np.allclose(fam.vertices(np.asarray(res['x'])), fam.vertices(q), atol=1e-8):
        raise ValueError(f"{record_name}: the last iterate's vertices are not result.x's")
    vv = fam.vertices(np.asarray(res['x']))
    out = {k: v2 for k, v2 in res.items() if k not in ('kind', 't', 'x')}
    out.update(
        schema=ledger.SCHEMA, N=head['N'], k=head['k'], start=head['start'],
        start_index=head['start'], convex=False, redraws=[], seeded=bool(head.get('seeded')),
        family='mirror', axis=fam.axis, chart=('fourier' if fourier else 'vertex'),
        # the evaluator's FINAL precision, as a real row has it (`spec.provenance`); the head
        # holds only the starting one. The last iterate's agrees on 57 of the 58 F2 checks.
        prec=it.get('prec', head.get('prec')), backend=head.get('backend'),
        douse_sha=head.get('douse_sha'), chladni_sha=head.get('chladni_sha'),
        lock_sha=head.get('lock_sha'), spec=head.get('spec'), spec_sha=head.get('spec_sha'),
        x_mirror=[float(t) for t in res['x']],
        x=[float(t) for t in param_vector(vv, PERIM)],
        vertices=[[float(z.real), float(z.imag)] for z in vv],
        edges=[float(t) for t in np.sort(dg.edge_lengths(vv))],
        angles=[float(t) for t in np.sort(np.degrees(dg.interior_angles(vv)))],
        seconds=None, n_resumes=None,
        antisym=dict(skipped=True, verdict='not run: salvaged from the run record; '
                                           'the probe did not finish'),
        salvaged={'from': record_name, 'by': 'benchmarks/salvage_record.py',
                  'at': datetime.datetime.now().isoformat(timespec='seconds')})
    if fourier:
        out['s'] = float(head['s'])
    return ledger._clean(out)          # nan as None, as `ledger.result_row` writes it


def find_records(root):
    return sorted(p for p in pathlib.Path(root).rglob('task_*.jsonl.start*.record.jsonl')
                  if RECORD.search(p.name))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='a run directory (cells below it) or one cell directory')
    ap.add_argument('--out-root', help='write ledgers here, same layout, instead of in place')
    ap.add_argument('--dry-run', action='store_true', help='list what would be written')
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    n_written = 0
    for rec in find_records(root):
        task, start = map(int, RECORD.search(rec.name).groups())
        led = rec.parent/f'task_{task}.jsonl'
        target = led if args.out_root is None else pathlib.Path(args.out_root)/led.relative_to(root)
        if any(r.get('start') == start for r in ledger.read_rows(led)):
            continue                                  # the run's own row is there
        if target != led and any(r.get('start') == start for r in ledger.read_rows(target)):
            continue
        with open(rec) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        try:
            row = salvage_row(rows, rec.name)
        except ValueError as exc:
            print(f'  SKIPPED {rec.relative_to(root)}: {exc}')
            continue
        if row is None:
            continue
        print(f"  {'would write' if args.dry_run else 'wrote'} {target}: start {start} "
              f"F = {row['fun']:.10f} ({row.get('convergence')})")
        if not args.dry_run:
            os.makedirs(target.parent, exist_ok=True)
            ledger.write_row(str(target), row)
        n_written += 1
    print(f"{n_written} row(s) {'to salvage' if args.dry_run else 'salvaged'}")
    return n_written


if __name__ == '__main__':
    main()
