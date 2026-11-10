"""A run specification: the file a grid is run from, and the file every row cites.  (2026-10-15)

The thesis grid will run on a cluster, across sessions and code versions, and a cell is not
always written by one of them. A `RunSpec` is the whole recipe for a sweep -- the cells, the
start sampler's seed and count, the optimizer's budget and radius, the solver's precision and
backend, the constraint family -- read from a TOML file, so a row can say which spec wrote it
(`spec`, `spec_sha`) and a reader can refuse to pool rows from different recipes. The spec is
content-hashed: editing the file changes the hash, and `summarize_grid` reports the mix.

    spec = load("benchmarks/specs/thesis_ratio_v0.toml")
    for N, k in spec.cells: ...

The git SHAs of douse and chladni are provenance of the CODE, recorded beside the spec on each
row (`provenance`), not part of the spec: the same spec run on a later commit is the same
experiment, and the SHAs say whether the numbers may differ.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

__all__ = ['RunSpec', 'load', 'git_sha', 'provenance', 'run_dir', 'PACKAGE_ROOT', 'SCHEDULING']

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fields that say where a spec runs rather than what it computes; not hashed (`RunSpec.content`).
SCHEDULING = ('slurm_mem', 'slurm_time', 'slurm_partition', 'slurm_account')


@dataclass(frozen=True)
class RunSpec:
    """One sweep. `cells` are `(N, k)` with `k` 0-based (k=2 is lam_3/lam_1), as the driver
    takes it. `family` is 'all' (star-shaped starts, the edge floor and the closing budget),
    'convex' (Valtr starts, plus the convexity constraint; a separate ledger) or 'mirror'
    (mirror-symmetric polygons, `douse.polygon.MirrorPolygon`; needs `seeds`, a separate ledger).
    `n_eigs=None` means the driver's default, `k + 3`.

    `seeds` names a JSON file (relative to the package root) of explicit start polygons,
    `{"N<N>_k<k>": [{"vertices": [[x, y], ..], "note": ".."}, ..]}`: start `i` of a cell is its
    `i`-th seed instead of a random draw, and a cell's starts beyond its seeds are skipped. The
    file's CONTENT is hashed into the spec (`seeds_sha`), so editing the seeds is a new recipe."""
    name: str
    cells: tuple
    starts: int = 8
    seed0: int = 0
    max_iter: int = 400
    delta0: float = 0.05
    prec: float = 1e-10
    prec_final: object = None
    backend: str = 'chladni'
    family: str = 'all'
    n_eigs: object = None
    sigma_max: float = 1e-6
    perim: float = 4.0
    seeds: str = ''
    notes: str = ''
    # WHERE IT RUNS, NOT WHAT IS RUN: Slurm resources per task for `benchmarks/slurm.py`
    # (`slurm_mem = "4G"`, `slurm_time = "24:00:00"`, partition and account as ARC names them).
    # Left out of `content()` and so of the hash: the same recipe given more memory is the same
    # experiment, and a spec without them hashes exactly as it did before they existed.
    slurm_mem: object = field(default=None, compare=False)
    slurm_time: object = field(default=None, compare=False)
    slurm_partition: object = field(default=None, compare=False)
    slurm_account: object = field(default=None, compare=False)
    source: str = field(default='', compare=False)

    def __post_init__(self):
        cells = tuple((int(N), int(k)) for N, k in self.cells)
        object.__setattr__(self, 'cells', cells)
        if self.family not in ('all', 'convex', 'mirror'):
            raise ValueError(f"family must be 'all', 'convex' or 'mirror', got {self.family!r}")
        if self.family == 'mirror' and not self.seeds:
            raise ValueError("family 'mirror' starts from seeds: give `seeds`")
        if self.backend not in ('chladni', 'lappy'):
            raise ValueError(f"backend must be 'chladni' or 'lappy', got {self.backend!r}")
        for N, k in cells:
            if N < 3 or k < 1:
                raise ValueError(f"a cell needs N >= 3 and k >= 1, got {(N, k)}")
        for key in SCHEDULING:
            v = getattr(self, key)
            if v is not None and not isinstance(v, str):
                raise ValueError(f"{key} must be a string as sbatch takes it, got {v!r}")

    @property
    def convex(self):
        return self.family == 'convex'

    def content(self):
        """The spec's content as a canonical dict: everything but where it was read from and
        where it is scheduled (`SCHEDULING`)."""
        d = asdict(self)
        for key in ('source',) + SCHEDULING:
            d.pop(key)
        d['cells'] = [list(c) for c in self.cells]
        if d['prec_final'] is None:
            d.pop('prec_final')        # so a spec without it hashes as it did before it existed
        if not self.seeds:
            d.pop('seeds')             # so a spec without seeds hashes as it did before they existed
        else:
            d['seeds_sha'] = hashlib.sha256(self.seeds_path().read_bytes()).hexdigest()[:12]
        return d

    def seeds_path(self):
        return PACKAGE_ROOT/self.seeds

    def seed_polygons(self, N, k):
        """The cell's seeds as complex vertex arrays (`[]` without `seeds`)."""
        if not self.seeds:
            return []
        table = json.loads(self.seeds_path().read_text())
        return [np.array([complex(x, y) for x, y in s['vertices']])
                for s in table.get(f'N{N}_k{k}', [])]

    @property
    def sha(self):
        """A short content hash: the identity a row cites."""
        blob = json.dumps(self.content(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def ledger_name(self, N, k):
        return f"N{N}_k{k}{'' if self.family == 'all' else '_' + self.family}.jsonl"


def load(path):
    """A `RunSpec` from a TOML file. Unknown keys are an error: a typo in a spec must not
    silently fall back to a default."""
    path = pathlib.Path(path)
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)
    known = {f for f in RunSpec.__dataclass_fields__} - {'source'}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; known: {sorted(known)}")
    if 'name' not in raw:
        raw['name'] = path.stem
    return RunSpec(source=str(path), **raw)


def run_dir(*parts):
    """Where a driver's outputs go by default: `$DOUSE_RUNS/<parts>` when the variable is set,
    else `benchmarks/run/<parts>` (relative, resolved against the douse checkout as before).

    On every machine `DOUSE_RUNS` points at one directory OUTSIDE every repository (workstation
    `~/projects/thesis-runs`, cluster `~/projects/thesis/runs`), so checkouts, worktrees and
    releases all read and write one copy, and removing one removes no results. Left unset, paths
    stay relative to the checkout.  (2026-11-03, the cluster 2026-11-09)"""
    env = os.environ.get('DOUSE_RUNS')
    if env:
        return str(pathlib.Path(env).expanduser().joinpath(*parts))
    return str(pathlib.PurePosixPath('benchmarks', 'run', *parts))


@lru_cache(maxsize=8)
def git_sha(path):
    """The short HEAD SHA of the repository at `path`, `-dirty` if tracked files have changed, or
    None. Cached: `code_shas` reads it at process start, not when the row is written days later,
    by which time the checkout may have moved."""
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if not out:
            return None
        dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain',
                                '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out + ('-dirty' if dirty else '')
    except Exception:                                                 # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def code_shas():
    """`(douse_sha, chladni_sha, lock_sha)` for this process. Call it when a run starts."""
    import douse
    root = pathlib.Path(douse.__file__).resolve().parent.parent
    lock = root/'uv.lock'
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()[:12] if lock.exists() else None
    try:
        import chladni
        chladni_sha = git_sha(pathlib.Path(chladni.__file__).resolve().parents[2])
    except Exception:                                                 # noqa: BLE001
        chladni_sha = None
    return git_sha(root), chladni_sha, lock_sha


def provenance(ev=None, spec=None):
    """WHICH CODE AND WHICH RECIPE WROTE A ROW: the backend and its precision, the SHAs of douse
    and chladni, and the spec's name and content hash when the run came from one."""
    douse_sha, chladni_sha, lock_sha = code_shas()
    row = dict(backend=(getattr(ev, 'backend', None) or 'chladni'),
               prec=(None if ev is None else float(getattr(ev, 'prec', float('nan')))),
               douse_sha=douse_sha, chladni_sha=chladni_sha, lock_sha=lock_sha)
    if spec is not None:
        row.update(spec=spec.name, spec_sha=spec.sha)
    return row
