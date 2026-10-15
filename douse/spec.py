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

__all__ = ['RunSpec', 'load', 'git_sha', 'provenance', 'PACKAGE_ROOT']

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    """One sweep. `cells` are `(N, k)` with `k` 0-based (k=2 is lam_3/lam_1), as the driver
    takes it. `family` is 'all' (star-shaped starts, the edge floor and the closing budget) or
    'convex' (Valtr starts, plus the convexity constraint; a separate ledger). `n_eigs=None`
    means the driver's default, `k + 2`."""
    name: str
    cells: tuple
    starts: int = 8
    seed0: int = 0
    max_iter: int = 400
    delta0: float = 0.05
    prec: float = 1e-10
    backend: str = 'chladni'
    family: str = 'all'
    n_eigs: object = None
    sigma_max: float = 1e-6
    perim: float = 4.0
    notes: str = ''
    source: str = field(default='', compare=False)

    def __post_init__(self):
        cells = tuple((int(N), int(k)) for N, k in self.cells)
        object.__setattr__(self, 'cells', cells)
        if self.family not in ('all', 'convex'):
            raise ValueError(f"family must be 'all' or 'convex', got {self.family!r}")
        if self.backend not in ('chladni', 'lappy'):
            raise ValueError(f"backend must be 'chladni' or 'lappy', got {self.backend!r}")
        for N, k in cells:
            if N < 3 or k < 1:
                raise ValueError(f"a cell needs N >= 3 and k >= 1, got {(N, k)}")

    @property
    def convex(self):
        return self.family == 'convex'

    def content(self):
        """The spec's content as a canonical dict (everything but where it was read from)."""
        d = asdict(self)
        d.pop('source')
        d['cells'] = [list(c) for c in self.cells]
        return d

    @property
    def sha(self):
        """A short content hash: the identity a row cites."""
        blob = json.dumps(self.content(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def ledger_name(self, N, k):
        return f"N{N}_k{k}{'_convex' if self.convex else ''}.jsonl"


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


def git_sha(path):
    """The short HEAD SHA of the repository at `path`, or None."""
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:                                                 # noqa: BLE001
        return None


def provenance(ev=None, spec=None):
    """WHICH CODE AND WHICH RECIPE WROTE A ROW: the backend and its precision, the SHAs of douse
    and chladni, and the spec's name and content hash when the run came from one."""
    import douse
    row = dict(backend=(getattr(ev, 'backend', None) or 'chladni'),
               prec=(None if ev is None else float(getattr(ev, 'prec', float('nan')))),
               douse_sha=git_sha(pathlib.Path(douse.__file__).resolve().parent.parent))
    try:
        import chladni
        row['chladni_sha'] = git_sha(pathlib.Path(chladni.__file__).resolve().parents[2])
    except Exception:                                                 # noqa: BLE001
        row['chladni_sha'] = None
    if spec is not None:
        row.update(spec=spec.name, spec_sha=spec.sha)
    return row
