"""The first `n_eigs` Dirichlet eigenvalues over a grid on a 2-parameter family of polygons.  (2026-10-26)

    python -m benchmarks.family_grid points benchmarks/specs/triangle_grid_v0.toml
    python -m benchmarks.family_grid run    benchmarks/specs/triangle_grid_v0.toml --task 0 --ntasks 1 [--limit 3]
    python -m benchmarks.family_grid slurm  benchmarks/specs/triangle_grid_v0.toml [--dry-run]
    python -m benchmarks.family_grid collect benchmarks/specs/triangle_grid_v0.toml
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

SCALE IS NOT NORMALIZED IN THE SOLVE. Every family fixes one length (a unit base); rows carry
`area` and `perim`, and `collect`/`plot` report `lam*area` (or perimeter^2) so the scale is a
choice made at read time.
"""
import argparse
import hashlib
import json
import os
import pathlib
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

def solve_point(spec, pid, a, b):
    """One row: the cold solve at one grid point. Never raises."""
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
            _, rec = Eigenproblem(dom).plan_and_solve(spec.n_eigs, prec=spec.prec)
            s = rec.spectrum
            unf = np.asarray(s.unfolded, dtype=float)
            row.update(status='ok' if len(unf) >= spec.n_eigs and not s.declined else 'short',
                       lam=[float(L.lam) for L in s.levels], mult=[int(L.mult) for L in s.levels],
                       bound=[float(L.bound) for L in s.levels],
                       met=[bool(L.met) for L in s.levels],
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


# ---- slurm ------------------------------------------------------------------------------------

def sbatch_text(spec, spec_path, out, mem, time_, partition, account, runner):
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name=grid_{spec.name}',
             f'#SBATCH --array=0-{spec.tasks - 1}',
             '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=1',
             f'#SBATCH --mem={mem}',
             f'#SBATCH --time={time_}']
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    if account:
        lines.append(f'#SBATCH --account={account}')
    lines += [f'#SBATCH --output={out}/logs/task_%a_%A.out',
              '',
              f'# spec {spec.name} @ {spec.sha}: {spec.family}, h={spec.h}, '
              f'{len(points(spec))} point(s) over {spec.tasks} task(s).',
              '# Generated by benchmarks/family_grid.py; submit from the douse checkout.',
              'set -euo pipefail',
              'cd "${DOUSE_ROOT:-$SLURM_SUBMIT_DIR}"', '']
    lines += [f'export {v}=1' for v in BLAS_VARS]
    lines += ['',
              f'{runner} -m benchmarks.family_grid run {shlex.quote(spec_path)} '
              f'--task ${{SLURM_ARRAY_TASK_ID}} --ntasks {spec.tasks} --out {shlex.quote(out)}',
              '']
    return '\n'.join(lines)


# ---- collect / plot ---------------------------------------------------------------------------

def collect(spec, out):
    """Pool the task ledgers: `(rows by id, report text)`. Rows from another spec_sha are refused."""
    pts = points(spec)
    rows, foreign = {}, 0
    for path in sorted(pathlib.Path(out).glob('task_*.jsonl')):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('spec_sha') != spec.sha:
                    foreign += 1
                    continue
                rows[r['id']] = r
    st = {}
    for r in rows.values():
        st[r['status']] = st.get(r['status'], 0) + 1
    missing = [i for i in range(len(pts)) if i not in rows]
    defs = np.array([r.get('weyl_deficit', np.nan) for r in rows.values()], dtype=float)
    walls = np.array([r['wall'] for r in rows.values()], dtype=float)
    bad_def = int(np.count_nonzero(defs > 0.5))
    unmet = sum(1 for r in rows.values() if r['status'] == 'ok' and not all(r['met']))
    lines = [f'# {spec.name} @ {spec.sha}: {len(rows)}/{len(pts)} point(s) solved, '
             f'{len(missing)} missing, status {st}',
             f'  Weyl deficit > 0.5 (a mode probably missed): {bad_def}',
             f'  ok rows with an unmet level: {unmet}',
             f'  rows from another spec_sha, ignored: {foreign}']
    if walls.size:
        lines.append(f'  wall per point: median {np.median(walls):.1f}s, max {walls.max():.1f}s, '
                     f'total {walls.sum()/3600:.2f} CPU-h')
    return rows, '\n'.join(lines)


def to_arrays(spec, rows):
    """`params (n, 2)`, `lam (n, n_eigs)` (NaN where absent), `area`, `perim`, `ids`."""
    ids = sorted(i for i, r in rows.items() if r['status'] in ('ok', 'short'))
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
    for name in ('points', 'run', 'slurm', 'collect', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('spec')
        p.add_argument('--out', default=None,
                       help='ledger directory, relative to the douse checkout '
                            '(default benchmarks/run/family_grid/<spec name>)')
        if name == 'run':
            p.add_argument('--task', type=int, default=0)
            p.add_argument('--ntasks', type=int, default=1)
            p.add_argument('--limit', type=int, default=None, help='solve at most this many')
        if name == 'slurm':
            p.add_argument('--mem', default=None)
            p.add_argument('--time', default=None)
            p.add_argument('--partition', default=None)
            p.add_argument('--account', default=None)
            p.add_argument('--runner', default='uv run python')
            p.add_argument('--dry-run', action='store_true')
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
    text = sbatch_text(spec, spec_path, out, mem, time_, args.partition or spec.slurm_partition,
                       args.account or spec.slurm_account, args.runner)
    path = pathlib.Path(out)/f'{spec.name}.sbatch'
    if args.dry_run:
        print(text)
        return 0
    dest = path if path.is_absolute() else PACKAGE_ROOT/path
    (dest.parent/'logs').mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f'# wrote {path}. Submit by hand, from the douse checkout:\nsbatch {path}\n'
          f'# then: uv run python -m benchmarks.family_grid plot {spec_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
