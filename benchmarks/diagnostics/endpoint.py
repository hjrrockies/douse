"""The shared front of the 2026-10-15 stall-analysis tools: a ledger row to a point and an evaluator.

Each tool in this directory that starts from a ledger endpoint (`probe_stall`, `scan_step`,
`replay`, `fd_hessian`, `warm_grad`) takes `seed setting` -- the `start` and `setting` of a row
in a ledger, by default the overnight comparison ledger
`benchmarks/run/diagnostics/overnight_2026_10_15.jsonl` -- and builds what it needs here, from
`douse.ledger` (reading) and `douse.runs` (the one evaluator factory and the shape helpers).
Each tool's core is a function of `(fam, ev, cons, p, k, ...)` that returns records, so it can
run on a fake evaluator in the tests; the `main` beside it loads a row and prints.

The two warm paths of a run (2026-10-14): `windowed`, the default, where the optimizer predicts
each level's move and hands chladni per-level windows; and `prefix`, the warm solve searching
the whole prefix, forced by hiding `predicted_moves` from the optimizer (`PrefixEvaluator`).

Run every tool from the douse root as a module, e.g.
`uv run python -m benchmarks.diagnostics.probe_stall 0 prefix`.
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np

from douse import ledger, runs

OVERNIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run',
                         'diagnostics', 'overnight_2026_10_15.jsonl')


class NoPrediction:
    """Wrap a solved shape so the optimizer cannot predict moves: the prefix warm path."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in ('predicted_moves',):
            raise AttributeError(name)
        return getattr(self._inner, name)


class PrefixEvaluator:
    """An evaluator whose solved shapes hide `predicted_moves` (see `NoPrediction`)."""

    def __init__(self, inner):
        self._inner = inner

    def eigenvalues(self, p, **kw):
        lams, solved = self._inner.eigenvalues(p, **kw)
        return lams, NoPrediction(solved)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def warm_path(ev, path):
    """`ev` on the named warm path: 'windowed' (as is) or 'prefix'."""
    if path == 'prefix':
        return PrefixEvaluator(ev)
    if path == 'windowed':
        return ev
    raise ValueError(f"warm path must be 'prefix' or 'windowed', got {path!r}")


def ratio(lams, k):
    """The objective `lam_{k+1}/lam_1` (0-based `k`, as `maximize_ratio` takes it)."""
    return float(lams[k]/lams[0])


@dataclass
class Endpoint:
    """A point to diagnose and everything the tools need around it."""
    fam: object
    ev: object
    cons: object
    p: np.ndarray
    k: int
    row: dict


def load_row(seed, setting, path=OVERNIGHT):
    """The last row in the ledger at `path` with this `start` and `setting`."""
    rows = [r for r in ledger.read_rows(path, dedupe=False)
            if r.get('start') == seed and r.get('setting') == setting]
    if not rows:
        raise LookupError(f'no row with start={seed} setting={setting!r} in {path}')
    return rows[-1]


def from_row(row):
    """An `Endpoint` at the row's final point `x`, with a fresh evaluator (cold first solve)."""
    N, k = int(row['N']), int(row['k'])
    fam = runs.family_for(N)
    ev = runs.make_evaluator(N, k, fam=fam)
    return Endpoint(fam=fam, ev=ev, cons=runs.constraints_for(fam), p=np.array(row['x'], float),
                    k=k, row=row)


def parser(doc):
    """The common command line: `seed setting [--ledger PATH]`."""
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument('seed', type=int, help="the row's `start`")
    ap.add_argument('setting', help="the row's `setting`, e.g. prefix, windowed2, windowed_ss")
    ap.add_argument('--ledger', default=OVERNIGHT)
    return ap


def endpoint_from_args(args):
    return from_row(load_row(args.seed, args.setting, args.ledger))
