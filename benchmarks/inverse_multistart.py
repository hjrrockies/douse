"""The inverse problem as an experiment: recover polygons from their first K eigenvalues.  (2026-10-31)

    python -m benchmarks.inverse_multistart runs    benchmarks/specs/inverse_pilot_v1.toml
    python -m benchmarks.inverse_multistart run     benchmarks/specs/inverse_pilot_v1.toml --task 0 --ntasks 1 [--limit 2]
    python -m benchmarks.inverse_multistart slurm   benchmarks/specs/inverse_pilot_v1.toml [--dry-run]
    python -m benchmarks.inverse_multistart collect benchmarks/specs/inverse_pilot_v1.toml

ONE RUN IS ONE LEAST-SQUARES FIT. A case is a known polygon (a rectangle, whose spectrum is exact,
or a polygon given by its vertices, solved once by chladni at `target_prec`). Its first K
eigenvalues, optionally perturbed by relative Gaussian noise of size `sigma`, are the target;
`optimize.gauss_newton` fits `SpectralMatch` (the K - 1 ratios lam_1/lam_j, so the target's
SIZE is never used) over the perimeter-`perim` N-gons from a start shape, and the row records the
fit and how far the recovered shape is from the true one MODULO CONGRUENCE (`congruence_error`):
translation, rotation, reflection and the cyclic relabelling of vertices all leave the spectrum
unchanged, so none of them may count as an error. Scale is fixed by the perimeter.

THE RUNS ARE A FUNCTION OF THE SPEC (`runs`): case x K x sigma x noise draw x start, in that
order, and a run's `id` is its index. `sigma = 0` gets one draw (there is nothing to draw).
Starts are `douse.runs.start_shape(N, i)` -- the same shapes for every K and sigma of a case,
so a row for K = 3 and one for K = 6 differ only in what was asked of the fit.

DETERMINED ONLY WHEN K - 1 >= 2N - 4 (`SpectralMatch.residual`): the family has 2N - 4
parameters, so a triangle needs 3 eigenvalues, a quadrilateral 5 and a pentagon 7. A spec may
ask for fewer -- an underdetermined fit converges onto a manifold of shapes, which is a result
too -- and the row says `determined`.

ONE LEDGER PER TASK, `<out>/task_<i>.jsonl`, task `i` of `T` running ids `i, i+T, ...`, resumed
by id like `family_grid`. A run that raises is a row with `status: error`, never dropped.
"""
import argparse
import hashlib
import json
import pathlib
import shlex
import sys
import time
import tomllib
import traceback
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from douse import spec as spec_mod

PACKAGE_ROOT = spec_mod.PACKAGE_ROOT
SCHEDULING = spec_mod.SCHEDULING
BLAS_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
# A start counts as a RECOVERY when its shape error is below this (vertex RMS at perimeter 1).
RECOVERED = 1e-3
# Two fits found the SAME shape when their congruence distance is below this. With noise the
# least-squares shape is not the true one, so recovery is judged against the best fit too.
SAME_BASIN = 1e-4
# Start shapes drawn in place of one the solver refuses, before the start counts as an error.
MAX_REDRAWS = 3


# ---- spec -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InverseSpec:
    """`cases` is a list of tables: `{name, kind = "rectangle", L, H}` or
    `{name, kind = "polygon", vertices = [[x, y], ...]}`, each with its own list `K`."""
    name: str
    cases: tuple
    sigmas: tuple = (0.0,)
    noise_draws: int = 1
    starts: int = 4
    seed0: int = 0
    max_iter: int = 40
    max_step: float = 0.1
    rtol: float = 1e-19
    prec: float = 1e-10
    target_prec: float = 1e-11
    perim: float = 4.0
    tasks: int = 100
    notes: str = ''
    slurm_mem: str | None = None
    slurm_time: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    source: str = field(default='', compare=False)

    def __post_init__(self):
        cases = []
        for c in self.cases:
            c = dict(c)
            if c.get('kind') not in ('rectangle', 'polygon'):
                raise ValueError(f"case {c.get('name')!r}: kind must be 'rectangle' or 'polygon'")
            if 'name' not in c or not c.get('K'):
                raise ValueError(f'every case needs a name and a list K, got {c!r}')
            c['K'] = [int(k) for k in c['K']]
            if min(c['K']) < 2:
                raise ValueError(f"case {c['name']!r}: K >= 2 (one eigenvalue is only a scale)")
            cases.append(c)
        if len({c['name'] for c in cases}) != len(cases):
            raise ValueError('case names must be unique')
        object.__setattr__(self, 'cases', tuple(cases))
        object.__setattr__(self, 'sigmas', tuple(float(s) for s in self.sigmas))

    def content(self):
        """What the rows depend on: everything but scheduling, notes, tasks and the file's path."""
        skip = set(SCHEDULING) | {'source', 'notes', 'tasks'}
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in skip}
        d['cases'] = [dict(c) for c in self.cases]
        d['sigmas'] = list(self.sigmas)
        return d

    @property
    def sha(self):
        blob = json.dumps(self.content(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load(path):
    path = pathlib.Path(path)
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)
    known = set(InverseSpec.__dataclass_fields__) - {'source'}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f'{path}: unknown key(s) {unknown}; known: {sorted(known)}')
    raw.setdefault('name', path.stem)
    return InverseSpec(source=str(path), **raw)


def default_out(spec):
    return f'benchmarks/run/inverse/{spec.name}'


# ---- cases and runs ---------------------------------------------------------------------------

def _perimeter(v):
    return float(np.abs(np.diff(np.append(v, v[0]))).sum())


def _signed_area(v):
    return 0.5*float(np.imag(np.sum(np.conj(v)*np.roll(v, -1))))


def case_vertices(case, perim):
    """The case's polygon, counter-clockwise, scaled to perimeter `perim`."""
    if case['kind'] == 'rectangle':
        L, H = float(case['L']), float(case['H'])
        v = np.array([0, L, L + 1j*H, 1j*H])
    else:
        v = np.array([complex(x, y) for x, y in case['vertices']])
    if _signed_area(v) < 0:
        v = v[::-1]
    return v*perim/_perimeter(v)


def rectangle_spectrum(L, H, n):
    """pi^2 (m^2/L^2 + k^2/H^2), the smallest `n` of them, exactly as floats."""
    M = n + 2
    vals = sorted(np.pi**2*(m**2/L**2 + k**2/H**2) for m in range(1, M) for k in range(1, M))
    return np.array(vals[:n])


def _solve_target(v, n, prec):
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    ep = Eigenproblem(Polygon(np.asarray(v), bc='dir'), coarse_bracket=True)
    _, rec = ep.plan_and_solve(n, prec=prec)
    s = rec.spectrum
    lam = np.asarray(s.unfolded, dtype=float)[:n]
    if len(lam) < n:
        raise RuntimeError(f'target solve returned {len(lam)} of {n} eigenvalues')
    worst = max(float(L.bound)/float(L.lam) for L in s.levels)
    return lam, worst


@lru_cache(maxsize=None)
def _target_cached(key, n, prec):
    case = json.loads(key)
    v = case_vertices(case, case['_perim'])
    if case['kind'] == 'rectangle':
        L, H = abs(v[1] - v[0]), abs(v[3] - v[0])
        return rectangle_spectrum(L, H, n), 0.0
    return _solve_target(v, n, prec)


def target_spectrum(spec, case):
    """`(lam, worst relative bound)`: the case's first max(K) eigenvalues at perimeter `perim`.
    Exact for a rectangle (bound 0); one chladni solve at `target_prec` otherwise, cached per
    process."""
    key = json.dumps(dict(case, _perim=spec.perim), sort_keys=True)
    return _target_cached(key, max(case['K']), spec.target_prec)


def runs(spec):
    """`[(case index, K, sigma, draw, start)]` in id order."""
    out = []
    for ci, case in enumerate(spec.cases):
        for K in case['K']:
            for sigma in spec.sigmas:
                for draw in range(1 if sigma == 0 else spec.noise_draws):
                    for start in range(spec.starts):
                        out.append((ci, K, sigma, draw, start))
    return out


def noisy(lam, sigma, seed):
    """`lam * (1 + sigma * e)`, `e` standard normal from `seed`; `sigma = 0` returns `lam`."""
    if sigma == 0:
        return np.array(lam, dtype=float)
    e = np.random.default_rng(seed).normal(size=len(lam))
    return np.asarray(lam)*(1 + sigma*e)


def _noise_seed(spec, ci, K, si, draw):
    return spec.seed0 + 1_000_003*ci + 10_007*K + 101*si + draw


# ---- shape error ------------------------------------------------------------------------------

def congruence_error(z, w):
    """RMS vertex distance between polygons `z` and `w` (same N), minimized over translation,
    rotation, reflection and the cyclic relabelling of vertices, divided by `w`'s perimeter.

    Each relabelling is a complex Procrustes problem: with both centred at their vertex means,
    `min_theta ||e^{i theta} z - w||` is attained at `theta = arg(sum(conj(z) w))`. A reflection
    is `conj` with the order reversed (to stay counter-clockwise). Returns `(error, matched z)`."""
    z = np.asarray(z, dtype=complex)
    w = np.asarray(w, dtype=complex)
    w0 = w - w.mean()
    best = (np.inf, None)
    for zz in (z, np.conj(z)[::-1]):
        zc = zz - zz.mean()
        for s in range(len(z)):
            zs = np.roll(zc, s)
            rot = np.sum(np.conj(zs)*w0)
            zr = zs*(rot/abs(rot) if abs(rot) > 0 else 1.0)
            e = float(np.sqrt(np.mean(np.abs(zr - w0)**2)))
            if e < best[0]:
                best = (e, zr + w.mean())
    return best[0]/_perimeter(w), best[1]


def _angles_deg(v):
    prev, nxt = np.roll(v, 1) - v, np.roll(v, -1) - v
    return np.degrees(np.mod(np.angle(prev/nxt), 2*np.pi))


# ---- one run ----------------------------------------------------------------------------------

def run_one(spec, rid, ci, K, sigma, draw, start):
    """One row: a Gauss-Newton fit to the case's (noisy) first K eigenvalues. Never raises."""
    from douse import SpectrumEvaluator, VertexPolygon, param_vector
    from douse.errors import ShapeRefusal
    from douse.objective import SpectralMatch
    from douse.optimize import gauss_newton
    from douse.runs import start_shape

    case = spec.cases[ci]
    w = case_vertices(case, spec.perim)
    N = len(w)
    si = spec.sigmas.index(sigma)
    row = dict(id=rid, case=case['name'], kind=case['kind'], N=N, K=K, sigma=sigma, draw=draw,
               start=start, determined=bool(K - 1 >= 2*N - 4), spec=spec.name,
               spec_sha=spec.sha)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            lam, tbound = target_spectrum(spec, case)
            lam = lam[:K]
            target = noisy(lam, sigma, _noise_seed(spec, ci, K, si, draw))
            row.update(target_lam=lam.tolist(), target=target.tolist(), target_bound=tbound,
                       target_min_gap=float(np.min(np.diff(lam)/lam[1:])))
            fam = VertexPolygon(N, perim=spec.perim)
            ev = SpectrumEvaluator(fam, n_eigs=K, prec=spec.prec, degenerate_tol=0.0,
                                   warn_gap=0.0)
            # A START THE SOLVER REFUSES IS REDRAWN, as `ratio_multistart` does: `gauss_newton`
            # treats a refusal at a trial step as a rejected step but cannot begin from a shape
            # with no spectrum (2026-10-31, tri_acute start 1: a random triangle missed 1e-10 on
            # all 3 levels). Redraw j uses start index `start + 1000 j`.
            redraws = []
            for j in range(MAX_REDRAWS + 1):
                p0 = param_vector(start_shape(N, start + 1000*j, spec.seed0, spec.perim),
                                  spec.perim)
                try:
                    ev(p0)
                    break
                except ShapeRefusal as exc:
                    redraws.append(f'{type(exc).__name__}: {str(exc)[:160]}')
                    ev.reset_continuation()
            else:
                raise ShapeRefusal(f'{len(redraws)} start shapes refused; last: {redraws[-1]}')
            row['redraws'] = redraws
            z0 = fam.vertices(p0)
            res = gauss_newton(ev, SpectralMatch(target), p0, max_iter=spec.max_iter,
                               max_step=spec.max_step, rtol=spec.rtol)
            z = fam.vertices(res.x)
            err, zm = congruence_error(z, w)
            err0, _ = congruence_error(z0, w)
            # how well the recovered shape reproduces the NOISELESS ratios: the fit's own
            # residual is against the noisy target
            try:
                fit = np.asarray(ev.eigenvalues(res.x)[0])[:K]
                ratio_err = float(np.max(np.abs(fit[0]/fit[1:] - lam[0]/lam[1:])))
            except ShapeRefusal:
                ratio_err = float('nan')
            row.update(status='ok', success=bool(res.success), message=str(res.message),
                       loss=float(res.fun), nit=int(res.nit), nfev=int(res.nfev),
                       n_refused=int(res.n_refused), grad_norm=float(res.grad_norm),
                       shape_err=err, start_err=err0, ratio_err=ratio_err,
                       recovered=bool(err < RECOVERED),
                       x=res.x.tolist(),
                       vertices=[[float(c.real), float(c.imag)] for c in zm],
                       start_vertices=[[float(c.real), float(c.imag)] for c in z0],
                       angles_deg=np.sort(_angles_deg(zm)).tolist(),
                       true_angles_deg=np.sort(_angles_deg(w)).tolist(),
                       worst_sigma=float(getattr(ev, 'worst_sigma', np.nan)))
        except Exception as exc:                                      # noqa: BLE001
            row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc().strip().splitlines()[-3:])
    row['wall'] = time.perf_counter() - t0
    row['warnings'] = sorted({f'{w_.category.__name__}: {w_.message}' for w_ in caught})[:20]
    return row


def _done_ids(path):
    if not path.exists():
        return set()
    with open(path) as fh:
        return {json.loads(line)['id'] for line in fh if line.strip()}


def run(spec, task, ntasks, out, limit=None):
    todo_all = runs(spec)
    mine = list(range(task, len(todo_all), ntasks))
    path = pathlib.Path(out)/f'task_{task}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(path)
    todo = [i for i in mine if i not in done]
    if limit is not None:
        todo = todo[:limit]
    prov = spec_mod.provenance()
    prov['prec'] = spec.prec
    print(f'# {spec.name} @ {spec.sha}: task {task}/{ntasks}, {len(mine)} run(s), '
          f'{len(done)} done, {len(todo)} to run -> {path}', flush=True)
    for i in todo:
        row = run_one(spec, i, *todo_all[i])
        row.update(prov, task=task, ntasks=ntasks)
        with open(path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(f"  id {i:4d} {row['case']:>12s} K={row['K']:2d} sigma={row['sigma']:.0e} "
              f"start {row['start']} {row['status']:5s} err {row.get('shape_err', np.nan):.2e} "
              f"(from {row.get('start_err', np.nan):.2e}) loss {row.get('loss', np.nan):.1e} "
              f"nit {row.get('nit', -1)} {row['wall']:7.1f}s", flush=True)


# ---- collect ----------------------------------------------------------------------------------

def _load(spec, out):
    rows, foreign = {}, 0
    for f in sorted(pathlib.Path(out).glob('task_*.jsonl')):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('spec_sha') != spec.sha:
                    foreign += 1
                    continue
                rows[r['id']] = r
    return rows, foreign


def summarize(spec, rows):
    """Per (case, K, sigma): runs, errors, recovery rate over starts and noise draws, how many
    starts reached the same shape as the BEST-FITTING start of their draw (least loss -- what a
    user without the answer would pick; `SAME_BASIN`), and that start's shape error, as median
    and max over draws."""
    groups = {}
    for r in rows.values():
        groups.setdefault((r['case'], r['K'], r['sigma']), []).append(r)
    out = []
    order = {c['name']: i for i, c in enumerate(spec.cases)}
    for key in sorted(groups, key=lambda k: (order[k[0]], k[1], k[2])):
        rs = groups[key]
        ok = [r for r in rs if r['status'] == 'ok']
        best, same = [], 0
        for d in sorted({r['draw'] for r in ok}):
            dr = [r for r in ok if r['draw'] == d]
            b = min(dr, key=lambda r: r['loss'])
            best.append(b['shape_err'])
            if 'vertices' in b:
                zb = np.array([complex(*c) for c in b['vertices']])
                same += sum(congruence_error(np.array([complex(*c) for c in r['vertices']]),
                                             zb)[0] < SAME_BASIN for r in dr)
        out.append(dict(case=key[0], N=rs[0]['N'], K=key[1], sigma=key[2], runs=len(rs),
                        errors=len(rs) - len(ok), determined=rs[0]['determined'],
                        recovered=sum(r['recovered'] for r in ok), same_basin=same,
                        best_err_median=float(np.median(best)) if best else np.nan,
                        best_err_max=float(np.max(best)) if best else np.nan,
                        wall_median=float(np.median([r['wall'] for r in rs]))))
    return out


def collect(spec, out):
    rows, foreign = _load(spec, out)
    n = len(runs(spec))
    lines = [f'# {spec.name} @ {spec.sha}: {len(rows)}/{n} run(s), {n - len(rows)} missing, '
             f'{foreign} row(s) from another spec_sha ignored',
             f"  {'case':>12s}  N   K  {'sigma':>7s}  runs  err  recovered  same  "
             f"{'best err (med)':>14s}  {'(max)':>8s}  wall med"]
    for s in summarize(spec, rows):
        lines.append(f"  {s['case']:>12s}  {s['N']}  {s['K']:2d}  {s['sigma']:7.0e}  "
                     f"{s['runs']:4d}  {s['errors']:3d}  {s['recovered']:4d}/{s['runs'] - s['errors']:<4d} "
                     f"{s['same_basin']:4d}  "
                     f"{s['best_err_median']:14.2e}  {s['best_err_max']:8.2e}  "
                     f"{s['wall_median']:7.0f}s{'' if s['determined'] else '  (underdetermined)'}")
    walls = [r['wall'] for r in rows.values()]
    if walls:
        lines.append(f'  wall per run: median {np.median(walls):.0f}s, max {max(walls):.0f}s, '
                     f'total {sum(walls)/3600:.2f} CPU-h')
    return rows, '\n'.join(lines)


# ---- slurm ------------------------------------------------------------------------------------

def sbatch_text(spec, spec_path, out, mem, time_, partition, account, runner, ntasks=None,
                mail=None):
    from benchmarks.slurm import mail_lines
    ntasks = ntasks or spec.tasks
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name=inverse_{spec.name}',
             f'#SBATCH --array=0-{ntasks - 1}',
             '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=1',
             f'#SBATCH --mem={mem}',
             f'#SBATCH --time={time_}']
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    if account:
        lines.append(f'#SBATCH --account={account}')
    lines += mail_lines(mail)
    lines += [f'#SBATCH --output={out}/logs/task_%a_%A.out',
              '',
              f'# spec {spec.name} @ {spec.sha}: {len(runs(spec))} run(s) over {ntasks} task(s).',
              '# Generated by benchmarks/inverse_multistart.py; submit from the douse checkout.',
              'set -euo pipefail',
              'cd "${DOUSE_ROOT:-$SLURM_SUBMIT_DIR}"', '']
    lines += [f'export {v}=1' for v in BLAS_VARS]
    lines += ['',
              f'{runner} -m benchmarks.inverse_multistart run {shlex.quote(spec_path)} '
              f'--task ${{SLURM_ARRAY_TASK_ID}} --ntasks {ntasks} --out {shlex.quote(out)}',
              '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('runs', 'run', 'slurm', 'collect'):
        p = sub.add_parser(name)
        p.add_argument('spec')
        p.add_argument('--out', default=None, help='default: benchmarks/run/inverse/<spec name>')
        if name == 'run':
            p.add_argument('--task', type=int, default=0)
            p.add_argument('--ntasks', type=int, default=1)
            p.add_argument('--limit', type=int, default=None, help='run at most this many')
        if name == 'slurm':
            p.add_argument('--mem', default=None)
            p.add_argument('--time', default=None)
            p.add_argument('--partition', default=None)
            p.add_argument('--account', default=None)
            p.add_argument('--mail', default=None,
                           help='mail END/FAIL notices here (default: $DOUSE_SLURM_MAIL)')
            p.add_argument('--runner', default='uv run python')
            p.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    spec = load(args.spec)
    out = args.out or default_out(spec)

    if args.cmd == 'runs':
        rs = runs(spec)
        print(f'# {spec.name} @ {spec.sha}: {len(rs)} run(s)')
        for ci, case in enumerate(spec.cases):
            n = sum(1 for r in rs if r[0] == ci)
            N = len(case_vertices(case, spec.perim))
            print(f"  {case['name']:>12s}  N={N}  K={case['K']}  {n} run(s)")
    elif args.cmd == 'run':
        run(spec, args.task, args.ntasks, out, args.limit)
    elif args.cmd == 'collect':
        _, text = collect(spec, out)
        print(text)
    elif args.cmd == 'slurm':
        from benchmarks.slurm import mail_address
        root = PACKAGE_ROOT
        spec_path = str(pathlib.Path(args.spec))
        try:
            spec_path = str(pathlib.Path(args.spec).resolve().relative_to(root.resolve()))
        except ValueError:
            pass
        text = sbatch_text(spec, spec_path, out, args.mem or spec.slurm_mem or '2G',
                           args.time or spec.slurm_time or '12:00:00',
                           args.partition or spec.slurm_partition,
                           args.account or spec.slurm_account, args.runner,
                           mail=mail_address(args.mail))
        if args.dry_run:
            print(text)
            return
        path = root/out/f'{spec.name}.sbatch'
        path.parent.mkdir(parents=True, exist_ok=True)
        (root/out/'logs').mkdir(exist_ok=True)
        path.write_text(text)
        print(f'# wrote {path.relative_to(root)}. Submit by hand, from the douse checkout:')
        print(f'sbatch {path.relative_to(root)}')
        print(f'# then: uv run python -m benchmarks.inverse_multistart collect {spec_path}')


if __name__ == '__main__':
    sys.exit(main())
