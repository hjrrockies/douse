"""The first `n_eigs` Dirichlet eigenvalues over a grid on a 2-parameter family of polygons.  (2026-10-26)

    python -m benchmarks.family_grid points benchmarks/specs/triangle_grid_v0.toml
    python -m benchmarks.family_grid run    benchmarks/specs/triangle_grid_v0.toml --task 0 --ntasks 1 [--limit 3]
    python -m benchmarks.family_grid slurm  benchmarks/specs/triangle_grid_v0.toml [--dry-run]
    python -m benchmarks.family_grid collect benchmarks/specs/triangle_grid_v0.toml
    python -m benchmarks.family_grid repair  benchmarks/specs/triangle_grid_v0.toml [--task 0 --ntasks 1]
    python -m benchmarks.family_grid plot    benchmarks/specs/triangle_grid_v0.toml

A FORWARD SWEEP, NO OPTIMIZER. Each grid point is one cold `plan_and_solve(n_eigs)`; the row
keeps the solver's LEVELS (value, multiplicity, bound, met), the unfolded prefix, the Weyl
deficit (`douse.spectral_guard`: a missed mode shows up as a deficit near +1, which no residual
can see) and every warning chladni raised. A point that raises is written as `status: error`
with the exception, never dropped, and `collect` counts it.

THE GRID IS A FUNCTION OF THE SPEC. Points are a lattice of spacing `h` over the family's
parameter box, kept when they lie in the family's fundamental domain (one representative per
similarity class) and every interior angle is at least `min_angle_deg`. Point `id` is the index
in that list, so a task, a resubmission and `collect` all agree on it without a shared file.

ONE LEDGER PER TASK, `<out>/task_<i>.jsonl`, task `i` of `T` solving points `i, i+T, i+2T, ...`.
The stride interleaves the expensive points (small angles, which pile up at one edge of the
domain) across tasks instead of handing them all to the last one. Rerunning a task skips the
ids its file already has, so a killed task is resubmitted as is (`sbatch --array=i ...`).

SUSPECT ROWS ARE REPAIRED, NOT TRUSTED (`suspect`, `repair`). A row is suspect when chladni's
count audit fired, or when two adjacent SIMPLE levels lie within their own error bounds of each
other -- one eigenvalue reported twice, which shifts every level above it by one index (the
triangle pilot, 2026-10-30: apex (0.15, 0.20), 1729.019 twice, 1e-10 apart with bounds of 1e-5,
and lam_9..lam_15 all one slot off). `repair` re-solves only those rows with `extra` more
eigenvalues (which removed that duplicate) into `<out>/repair/`, and `collect`, `to_arrays` and
`plot` prefer a repaired row and leave out any row still suspect after its repair -- except
where the repair shows evidence of a defect and the original, re-solved only for the last-rung
precaution, shows none (`keeps_original`): then the original is kept, marked `original_kept`.

SCALE IS NOT NORMALIZED IN THE SOLVE. Every family fixes one length (a unit base); rows carry
`area` and `perim`, and `collect`/`plot` report `lam*area` (or perimeter^2) so the scale is a
choice made at read time.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import sys
import time
import tomllib
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np

from douse import spec as spec_mod

PACKAGE_ROOT = spec_mod.PACKAGE_ROOT
SCHEDULING = spec_mod.SCHEDULING
BLAS_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


# ---- families ---------------------------------------------------------------------------------
# Each: vertices(a, b) counter-clockwise, the parameter box, and the fundamental domain.

def _tri(a, b):
    return np.array([0, 1, a + 1j*b])


def _tri_ok(a, b):
    # Base [0, 1] is the longest side and the apex leans left: one triangle per similarity class.
    return 0 <= a <= 0.5 and b > 0 and (1 - a)**2 + b**2 <= 1 + 1e-12


def _para(a, b):
    return np.array([0, 1, 1 + a + 1j*b, a + 1j*b])


def _para_ok(a, b):
    # Unit base is the longer side, the slanted side leans right: one per similarity class.
    return a >= 0 and b > 0 and a**2 + b**2 <= 1 + 1e-12


def _trap(w, h):
    return np.array([-0.5, 0.5, w/2 + 1j*h, -w/2 + 1j*h])


def _trap_ok(w, h):
    # Isosceles trapezoid, unit bottom, top width w in (0, 1] (w = 1 is the rectangle).
    return 0 < w <= 1 and h > 0


def _kite(a, b):
    return np.array([0, a - 1j*b, 1, a + 1j*b])


def _kite_ok(a, b):
    # Convex kite with axis [0, 1]; a <= 1/2 up to reflection (a = 1/2 is the rhombus).
    return 0 < a <= 0.5 and b > 0


FAMILIES = {
    'triangle': dict(vertices=_tri, ok=_tri_ok, box=((0.0, 0.5), (0.0, np.sqrt(3)/2)),
                     params=('a', 'b')),
    'parallelogram': dict(vertices=_para, ok=_para_ok, box=((0.0, 1.0), (0.0, 1.0)),
                          params=('a', 'b')),
    'isosceles_trapezoid': dict(vertices=_trap, ok=_trap_ok, box=((0.0, 1.0), (0.0, 1.5)),
                                params=('w', 'h')),
    'kite': dict(vertices=_kite, ok=_kite_ok, box=((0.0, 0.5), (0.0, 1.0)), params=('a', 'b')),
}


def interior_angles_deg(v):
    """Interior angles of a counter-clockwise polygon, degrees."""
    prev, nxt = np.roll(v, 1) - v, np.roll(v, -1) - v
    return np.degrees(np.mod(np.angle(prev/nxt), 2*np.pi))


def area(v):
    x, y = v.real, v.imag
    return 0.5*float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def perimeter(v):
    return float(np.abs(np.diff(np.concatenate([v, v[:1]]))).sum())


# ---- spec -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    name: str
    family: str
    h: float
    n_eigs: int = 15
    prec: float = 1e-10
    min_angle_deg: float = 10.0
    coarse_bracket: bool = False
    tasks: int = 100
    notes: str = ''
    slurm_mem: str | None = None
    slurm_time: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    source: str = field(default='', compare=False)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f'unknown family {self.family!r}; known: {sorted(FAMILIES)}')

    def content(self):
        """What the rows depend on: everything but scheduling, notes and the file's path."""
        skip = set(SCHEDULING) | {'source', 'notes', 'tasks'}
        if not self.coarse_bracket:
            skip.add('coarse_bracket')        # specs written before the key keep their hash
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in skip}

    @property
    def sha(self):
        blob = json.dumps(self.content(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load(path):
    path = pathlib.Path(path)
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)
    known = set(GridSpec.__dataclass_fields__) - {'source'}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f'{path}: unknown key(s) {unknown}; known: {sorted(known)}')
    raw.setdefault('name', path.stem)
    return GridSpec(source=str(path), **raw)


def points(spec):
    """`[(a, b)]`, the grid in id order (row-major in `b`, then `a`)."""
    fam = FAMILIES[spec.family]
    (a0, a1), (b0, b1) = fam['box']
    n_a = int(round((a1 - a0)/spec.h))
    n_b = int(round((b1 - b0)/spec.h))
    out = []
    for b in b0 + spec.h*np.arange(n_b + 1):
        for a in a0 + spec.h*np.arange(n_a + 1):
            a, b = float(round(a, 12)), float(round(b, 12))
            if not fam['ok'](a, b):
                continue
            if interior_angles_deg(fam['vertices'](a, b)).min() < spec.min_angle_deg:
                continue
            out.append((a, b))
    return out


def default_out(spec):
    return f'benchmarks/run/family_grid/{spec.name}'


# ---- solve ------------------------------------------------------------------------------------

def _keep_levels(mults, n):
    """How many leading levels cover the first `n` eigenvalues (a multiple level that straddles
    `n` is kept whole)."""
    total = 0
    for i, m in enumerate(mults):
        if total >= n:
            return i
        total += m
    return len(mults)


def solve_point(spec, pid, a, b, n_solve=None):
    """One row: the cold solve at one grid point. Never raises.

    `n_solve` (a repair) solves for more eigenvalues than the spec stores; the row keeps the
    levels covering the first `n_eigs` and records `n_solved`."""
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    from douse.spectral_guard import weyl_deficit

    v = FAMILIES[spec.family]['vertices'](a, b)
    row = dict(id=pid, family=spec.family, a=a, b=b,
               vertices=[[float(z.real), float(z.imag)] for z in v],
               area=area(v), perim=perimeter(v),
               min_angle_deg=float(interior_angles_deg(v).min()),
               n_eigs=spec.n_eigs, spec=spec.name, spec_sha=spec.sha)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            dom = Polygon(v, bc='dir')
            ep = (Eigenproblem(dom, coarse_bracket=True) if spec.coarse_bracket
                  else Eigenproblem(dom))
            n = int(n_solve or spec.n_eigs)
            _, rec = ep.plan_and_solve(n, prec=spec.prec)
            s = rec.spectrum
            unf = np.asarray(s.unfolded, dtype=float)
            levels = s.levels[:_keep_levels([int(L.mult) for L in s.levels], spec.n_eigs)]
            if n != spec.n_eigs:
                row['n_solved'] = n
            row.update(status='ok' if len(unf) >= spec.n_eigs and not s.declined else 'short',
                       lam=[float(L.lam) for L in levels], mult=[int(L.mult) for L in levels],
                       bound=[float(L.bound) for L in levels],
                       met=[bool(L.met) for L in levels],
                       unfolded=unf[:spec.n_eigs].tolist(),
                       declined=bool(s.declined), reason=str(s.reason),
                       flags=[str(f) for f in s.flags],
                       weyl_deficit=float(weyl_deficit(dom, unf)))
        except Exception as exc:                                      # noqa: BLE001
            row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc().strip().splitlines()[-3:])
    row['wall'] = time.perf_counter() - t0
    row['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    return row


def _done_ids(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                done.add(json.loads(line)['id'])
    return done


def run(spec, task, ntasks, out, limit=None):
    pts = points(spec)
    mine = list(range(task, len(pts), ntasks))
    path = pathlib.Path(out)/f'task_{task}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(path)
    todo = [i for i in mine if i not in done]
    if limit is not None:
        todo = todo[:limit]
    prov = spec_mod.provenance()
    prov['prec'] = spec.prec
    print(f'# {spec.name} @ {spec.sha}: task {task}/{ntasks}, {len(mine)} point(s), '
          f'{len(done)} done, {len(todo)} to solve -> {path}', flush=True)
    for i in todo:
        a, b = pts[i]
        row = solve_point(spec, i, a, b)
        row.update(prov, task=task, ntasks=ntasks)
        with open(path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        first = row.get('unfolded', [float('nan')])[:1]
        print(f"  id {i:5d} ({a:.4f}, {b:.4f}) {row['status']:5s} {row['wall']:7.1f}s "
              f"lam1 {first[0] if first else float('nan'):.6f} "
              f"deficit {row.get('weyl_deficit', float('nan')):.2f}", flush=True)


# ---- suspect rows and their repair -------------------------------------------------------------

# A level whose error bound exceeds this fraction of its value was never located: a phantom.
# The full triangle grid (2026-10-30, 2305 rows) has NO row with a worst relative bound between
# 1e-5 and 1e-3 -- real levels sit at or below 3e-6, the phantoms at 0.3..18 -- so the cut is
# not a tuning knob.
UNRESOLVED_REL = 1e-4

_RUNG_ONLY = ['planner fell back to its last rung']

# chladni's planner flag when no rung met prec: '... returning rung 4 of 4, the one with ...'
_FALLBACK = re.compile(r'returning rung (\d+) of (\d+)')


def suspect(row):
    """Why a row's spectrum cannot be trusted as is: `[]` for a clean row.

    * `status` -- not `ok` (an error, a short spectrum);
    * `count audit` -- chladni's count audit fired (`short`: a level missing; `excess`: one
      counted twice). The worst-cut `weyl_deficit` is NOT used: on complete spectra it raises
      false alarms (four fat triangles in the pilot), which the mean-cut audit does not;
    * `overlapping levels` -- two adjacent levels, both reported SIMPLE, closer together than
      the sum of their error bounds. A real double comes back as one level of multiplicity 2,
      so two simple levels that the bounds cannot separate are one eigenvalue found twice;
    * `unresolved level` -- a level whose error bound exceeds `UNRESOLVED_REL` of its value: not
      an eigenvalue at all (full triangle grid: 10 rows, bounds 0.3..18 x the level, three of them
      invisible to the count audit, one a well-shaped 45-50-85 degree triangle);
    * `planner fell back to its last rung` -- no rung met `prec` and the planner returned its
      LAST one. On the full triangle grid that outcome was suspect by the rules above in 11 of
      18 rows (61%), against 0 of 1960 rows where a rung met `prec` and 2 of 327 that fell back
      to an earlier rung; re-solving the other 7 is cheaper than trusting their bounds. Not
      applied to a repaired row (`repair_of`) or a kept original (`keeps_original`): the re-solve is the second
      look, and it is judged on the evidence rules above only, so a clean repair is never
      dropped for its rung.
    """
    why = []
    if row.get('status') != 'ok':
        why.append(f"status {row.get('status')}")
    if any(str(f).startswith('count audit') for f in row.get('flags', ())):
        why.append('count audit')
    lam, mult, bound = row.get('lam') or [], row.get('mult') or [], row.get('bound') or []
    for i in range(len(lam) - 1):
        if mult[i] == 1 and mult[i + 1] == 1 and lam[i + 1] - lam[i] <= bound[i] + bound[i + 1]:
            why.append(f'overlapping levels {i + 1},{i + 2}')
            break
    for i in range(len(lam)):
        if bound[i] > UNRESOLVED_REL*abs(lam[i]):
            why.append(f'unresolved level {i + 1}')
            break
    second_look = row.get('repair_of') is not None or row.get('original_kept')
    for f in ([] if second_look else row.get('flags', ())):
        m = _FALLBACK.search(str(f))
        if m and m.group(1) == m.group(2):
            why.append('planner fell back to its last rung')
            break
    return why


def _load(spec, folder):
    """`(rows by id, foreign count)` from every `task_*.jsonl` in `folder`."""
    rows, foreign = {}, 0
    for path in sorted(pathlib.Path(folder).glob('task_*.jsonl')):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('spec_sha') != spec.sha:
                    foreign += 1
                    continue
                rows[r['id']] = r
    return rows, foreign


def repair(spec, out, task=0, ntasks=1, extra=5, limit=None):
    """Re-solve the suspect rows of the primary ledgers with `extra` more eigenvalues, into
    `<out>/repair/task_<task>.jsonl`. A row already repaired is not repeated, and a repair that
    is itself suspect is kept as written (collect leaves it out): no loop."""
    rows, _ = _load(spec, out)
    done = set(_load(spec, pathlib.Path(out)/'repair')[0])
    bad = sorted(i for i, r in rows.items() if suspect(r))
    mine = [i for i in bad[task::ntasks] if i not in done]
    if limit is not None:
        mine = mine[:limit]
    path = pathlib.Path(out)/'repair'/f'task_{task}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    prov = spec_mod.provenance()
    prov['prec'] = spec.prec
    print(f'# {spec.name} @ {spec.sha}: {len(bad)} suspect row(s), {len(done)} already repaired; '
          f'task {task}/{ntasks} re-solves {len(mine)} with n_eigs + {extra} -> {path}', flush=True)
    for i in mine:
        old = rows[i]
        row = solve_point(spec, i, old['a'], old['b'], n_solve=spec.n_eigs + extra)
        row.update(prov, task=task, ntasks=ntasks, repair_of=suspect(old))
        with open(path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(f"  id {i:5d} ({old['a']:.4f}, {old['b']:.4f}) was {suspect(old)} -> "
              f"{row['status']} {suspect(row) or 'clean'} {row['wall']:7.1f}s", flush=True)


# ---- slurm ------------------------------------------------------------------------------------

def sbatch_text(spec, spec_path, out, mem, time_, partition, account, runner, cmd='run',
                ntasks=None, mail=None):
    ntasks = ntasks or spec.tasks
    logs = f'{out}/logs' if cmd == 'run' else f'{out}/{cmd}/logs'
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name={"grid" if cmd == "run" else cmd}_{spec.name}',
             f'#SBATCH --array=0-{ntasks - 1}',
             '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=1',
             f'#SBATCH --mem={mem}',
             f'#SBATCH --time={time_}']
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    if account:
        lines.append(f'#SBATCH --account={account}')
    from benchmarks.slurm import mail_lines
    lines += mail_lines(mail)
    lines += [f'#SBATCH --output={logs}/task_%a_%A.out',
              '',
              (f'# spec {spec.name} @ {spec.sha}: {spec.family}, h={spec.h}, '
               f'{len(points(spec))} point(s) over {ntasks} task(s).' if cmd == 'run' else
               f'# spec {spec.name} @ {spec.sha}: re-solve the suspect rows over {ntasks} task(s).'),
              '# Generated by benchmarks/family_grid.py; submit from the douse checkout.',
              'set -euo pipefail',
              'cd "${DOUSE_ROOT:-$SLURM_SUBMIT_DIR}"', '']
    lines += [f'export {v}=1' for v in BLAS_VARS]
    lines += ['',
              f'{runner} -m benchmarks.family_grid {cmd} {shlex.quote(spec_path)} '
              f'--task ${{SLURM_ARRAY_TASK_ID}} --ntasks {ntasks} --out {shlex.quote(out)}',
              '']
    return '\n'.join(lines)


# ---- collect / plot ---------------------------------------------------------------------------

def keeps_original(repair_row, original):
    """Should `collect` keep the original row rather than its repair?

    When the repair shows EVIDENCE of a defect (any `suspect` reason: the last-rung rule is not
    applied to a repair) and the original shows none -- it was re-solved only for the last-rung
    precaution. Full triangle grid, 2026-10-30: all 7 rung-only originals were right. Five clean
    repairs agreed with them to <= 1.6e-8; id 2055's repair agreed to 7.6e-12 but read lam_15 and
    lam_16 (beyond the stored range) as an uncertified double; id 276's repair put a duplicate at
    levels 4-5 and was off by 6..18% above it, while the original matched its neighbours to 0.4%.
    A row without evidence beats a row with evidence; an original WITH evidence is never kept."""
    return suspect(original) == _RUNG_ONLY and bool(suspect(repair_row))


def collect(spec, out):
    """Pool the task ledgers: `(rows by id, report text)`. Rows from another spec_sha are refused;
    a row in `<out>/repair/` replaces the primary row for its id (marked `repaired`), unless
    `keeps_original` -- the repair shows evidence of a defect and the original none -- in which
    case the original is kept (marked `original_kept`)."""
    pts = points(spec)
    rows, foreign = _load(spec, out)
    flagged = sorted(i for i, r in rows.items() if suspect(r))
    repaired, f2 = _load(spec, pathlib.Path(out)/'repair')
    foreign += f2
    kept = []
    for i, r in repaired.items():
        if i in rows and keeps_original(r, rows[i]):
            rows[i] = dict(rows[i], original_kept=True)
            kept.append(i)
        else:
            rows[i] = dict(r, repaired=True)
    still = sorted(i for i, r in rows.items() if suspect(r))
    st = {}
    for r in rows.values():
        st[r['status']] = st.get(r['status'], 0) + 1
    missing = [i for i in range(len(pts)) if i not in rows]
    walls = np.array([r['wall'] for r in rows.values()], dtype=float)
    pending = [i for i in flagged if i not in repaired]
    unmet = sum(1 for r in rows.values() if r['status'] == 'ok' and not all(r['met']))
    lines = [f'# {spec.name} @ {spec.sha}: {len(rows)}/{len(pts)} point(s) solved, '
             f'{len(missing)} missing, status {st}',
             f'  suspect rows (count audit, duplicated or unresolved levels, last-rung fallback): '
             f'{len(flagged)}; repaired {len(repaired)}, awaiting repair {len(pending)}, '
             f'still suspect after repair {len([i for i in still if i in repaired])}'
             + (f' (originals kept over a repair with a defect: {len(kept)})' if kept else '')
             + (f' -> run `repair`' if pending else '')
             + (f'; left out of arrays and plots: ids {still[:12]}{"..." if len(still) > 12 else ""}'
                if still else ''),
             f'  ok rows with an unmet level: {unmet}',
             _bound_line(rows),
             f'  rows from another spec_sha, ignored: {foreign}']
    if walls.size:
        lines.append(f'  wall per point: median {np.median(walls):.1f}s, max {walls.max():.1f}s, '
                     f'total {walls.sum()/3600:.2f} CPU-h')
    return rows, '\n'.join(lines)


def _bound_line(rows):
    """The worst RELATIVE level bound per row (chladni's `bound` is absolute, Moler-Payne), as
    a distribution: the accuracy the grid actually has, which on flat shapes is short of `prec`
    and not flagged by the Weyl deficit (2026-10-26: 1e-8..1e-5 at height 0.10-0.15)."""
    worst = np.array([max(np.divide(r['bound'], r['lam'])) for r in rows.values()
                      if r.get('bound')], dtype=float)
    if not worst.size:
        return '  worst relative level bound per row: none'
    q = np.quantile(worst, [0.5, 0.9, 1.0])
    edges = (1e-9, 1e-7, 1e-5, 1e-3)
    counts = [int(np.count_nonzero(worst <= e)) for e in edges]
    return (f'  worst relative level bound per row: median {q[0]:.0e}, 90% {q[1]:.0e}, max {q[2]:.0e}; '
            + ', '.join(f'<= {e:.0e}: {c}' for e, c in zip(edges, counts)) + f' of {worst.size}')


def to_arrays(spec, rows):
    """`params (n, 2)`, `lam (n, n_eigs)` (NaN where absent), `area`, `perim`, `ids`. Rows still
    suspect (after any repair) are left out."""
    ids = sorted(i for i, r in rows.items() if r['status'] in ('ok', 'short') and not suspect(r))
    P = np.array([[rows[i]['a'], rows[i]['b']] for i in ids]).reshape(-1, 2)
    L = np.full((len(ids), spec.n_eigs), np.nan)
    for j, i in enumerate(ids):
        u = rows[i]['unfolded']
        L[j, :len(u)] = u
    A = np.array([rows[i]['area'] for i in ids])
    Pm = np.array([rows[i]['perim'] for i in ids])
    return P, L, A, Pm, np.array(ids)


def plot(spec, out, scale='area'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    rows, rep = collect(spec, out)
    print(rep)
    P, L, A, Pm, _ = to_arrays(spec, rows)
    if len(P) < 3:
        print('# fewer than 3 solved points; nothing to plot')
        return
    s = A if scale == 'area' else Pm**2
    tri = mtri.Triangulation(P[:, 0], P[:, 1])
    names = FAMILIES[spec.family]['params']
    figdir = pathlib.Path(out)/'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    n = spec.n_eigs
    cols = 5
    fig, axes = plt.subplots(-(-n//cols), cols, figsize=(3.2*cols, 2.6*-(-n//cols)),
                             constrained_layout=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis('off')
            continue
        z = L[:, k]*s
        ok = np.isfinite(z)
        cs = ax.tricontourf(P[ok, 0], P[ok, 1], z[ok], levels=20, cmap='viridis')
        ax.tricontour(P[ok, 0], P[ok, 1], z[ok], levels=20, colors='k', linewidths=0.3)
        fig.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_title(f'$\\lambda_{{{k + 1}}}\\cdot${"A" if scale == "area" else "P$^2$"}')
        ax.set_aspect('equal')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
    fig.suptitle(f'{spec.name}: {spec.family}, {len(P)} points')
    p1 = figdir/f'lam_times_{scale}.png'
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # Gaps relative to lam_1: where the spectrum nearly degenerates (the hard part of optimization).
    fig, axes = plt.subplots(-(-(n - 1)//cols), cols, figsize=(3.2*cols, 2.6*-(-(n - 1)//cols)),
                             constrained_layout=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n - 1:
            ax.axis('off')
            continue
        g = np.log10(np.abs(L[:, k + 1] - L[:, k])/L[:, 0])
        ok = np.isfinite(g)
        cs = ax.tricontourf(P[ok, 0], P[ok, 1], g[ok], levels=20, cmap='magma')
        fig.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_title(f'log10 ($\\lambda_{{{k + 2}}}-\\lambda_{{{k + 1}}})/\\lambda_1$')
        ax.set_aspect('equal')
    p2 = figdir/'gaps.png'
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    np.savez(pathlib.Path(out)/'grid.npz', params=P, lam=L, area=A, perim=Pm,
             spec=spec.name, spec_sha=spec.sha)
    print(f'# wrote {p1}, {p2}, {pathlib.Path(out)/"grid.npz"}')
    del tri


# ---- cli --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('points', 'run', 'slurm', 'collect', 'repair', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('spec')
        p.add_argument('--out', default=None,
                       help='ledger directory, relative to the douse checkout '
                            '(default benchmarks/run/family_grid/<spec name>)')
        if name in ('run', 'repair'):
            p.add_argument('--task', type=int, default=0)
            p.add_argument('--ntasks', type=int, default=1)
            p.add_argument('--limit', type=int, default=None, help='solve at most this many')
        if name == 'repair':
            p.add_argument('--extra', type=int, default=5,
                           help='eigenvalues to solve beyond n_eigs (default 5)')
        if name == 'slurm':
            p.add_argument('--mem', default=None)
            p.add_argument('--time', default=None)
            p.add_argument('--partition', default=None)
            p.add_argument('--account', default=None)
            p.add_argument('--mail', default=None,
                           help='mail END/FAIL notices here (default: $DOUSE_SLURM_MAIL, else none)')
            p.add_argument('--runner', default='uv run python')
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--repair', type=int, default=None, metavar='NTASKS',
                           help='write the repair job instead, over NTASKS array tasks')
        if name == 'plot':
            p.add_argument('--scale', choices=('area', 'perim'), default='area')
    args = ap.parse_args(argv)
    spec = load(args.spec)
    out = args.out or default_out(spec)

    if args.cmd == 'points':
        pts = points(spec)
        print(f'# {spec.name} @ {spec.sha}: {spec.family}, h={spec.h}, min angle '
              f'{spec.min_angle_deg} deg -> {len(pts)} point(s), {spec.tasks} task(s), '
              f'~{-(-len(pts)//spec.tasks)} per task')
        return 0
    if args.cmd == 'run':
        run(spec, args.task, args.ntasks, out, args.limit)
        return 0
    if args.cmd == 'repair':
        repair(spec, out, args.task, args.ntasks, args.extra, args.limit)
        return 0
    if args.cmd == 'collect':
        print(collect(spec, out)[1])
        return 0
    if args.cmd == 'plot':
        plot(spec, out, args.scale)
        return 0

    # slurm
    mem = args.mem or spec.slurm_mem or '4G'
    time_ = args.time or spec.slurm_time or '24:00:00'
    spec_path = str(pathlib.Path(args.spec).resolve().relative_to(PACKAGE_ROOT)) \
        if pathlib.Path(args.spec).resolve().is_relative_to(PACKAGE_ROOT) else args.spec
    from benchmarks.slurm import mail_address
    cmd = 'repair' if args.repair else 'run'
    text = sbatch_text(spec, spec_path, out, mem, time_, args.partition or spec.slurm_partition,
                       args.account or spec.slurm_account, args.runner, cmd=cmd,
                       ntasks=args.repair, mail=mail_address(args.mail))
    path = (pathlib.Path(out)/f'{spec.name}.sbatch' if cmd == 'run'
            else pathlib.Path(out)/'repair'/f'{spec.name}_repair.sbatch')
    if args.dry_run:
        print(text)
        return 0
    dest = path if path.is_absolute() else PACKAGE_ROOT/path
    (dest.parent/'logs').mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f'# wrote {path}. Submit by hand, from the douse checkout:\nsbatch {path}\n'
          f'# then: uv run python -m benchmarks.family_grid '
          f'{"plot" if cmd == "run" else "collect"} {spec_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
