"""The Bayesian triangle posterior from noisy eigenvalue ratios, computed on a grid.  (2026-11-03)

    python -m benchmarks.triangle_posterior prepare benchmarks/specs/triangle_posterior_v1.toml
    python -m benchmarks.triangle_posterior run     benchmarks/specs/triangle_posterior_v1.toml --task 0 --ntasks 1 [--limit 2]
    python -m benchmarks.triangle_posterior slurm   benchmarks/specs/triangle_posterior_v1.toml [--dry-run]
    python -m benchmarks.triangle_posterior collect benchmarks/specs/triangle_posterior_v1.toml
    python -m benchmarks.triangle_posterior plot    benchmarks/specs/triangle_posterior_v1.toml

NO SOLVES. The forward map is the triangle grid (`family_grid`, `triangle_grid_v1`: lam_1..15 at
2461 apex positions a + ib, base [0, 1], every angle >= 10 deg), interpolated. The data are the
ratios nu_j = lam_1/lam_(j+1), j = 1..K-1, with iid Gaussian noise of unknown size sigma.

THE MODEL. Shape space is the grid's support in apex coordinates (one triangle per similarity
class, so the mirror ambiguity is already quotiented out). The prior is Kendall's uniform shape
measure times exp(-beta (E - E_eq)), E the tangent-point energy (`douse.repulsive`), truncated to
the support. The likelihood is N(nu(x), (sigma^2 + tau_j^2) I): tau_j is the interpolant's own
cross-validated error, so model error is not mistaken for information. sigma^2 ~ InvGamma(
ig_alpha, ig_beta) and is integrated out on a log grid (`sigma_grid`). `ig_beta = "matched"`
puts the prior's mode at the true sigma^2, beta = (alpha + 1) sigma^2: sigma known to within a
factor of a few. A VAGUE prior does not calibrate (trial, 2026-11-03, IG(1, 1e-7)): at K = 3 two
ratios meet two shape parameters, other triangles fit the data exactly, and the sigma marginal
prefers a tiny sigma at a wrong exact fit; with beta = 1 and an obtuse truth, the heavy upper tail
makes a large sigma cheaper than the energy penalty, and the posterior explains the data as noise.

NO INVERSE CRIME. Grid points split into 4 folds by lattice parity, (i mod 2) + 2 (j mod 2). Fold
f's interpolant is fitted to the OTHER three folds, and a truth from fold f is scored only with
it: the data are the truth's solved eigenvalues, never values of the interpolant used to invert
them. Truths are interior grid points (all four lattice neighbours present): at the support's
edge the cross-validation extrapolates and its error is up to 7e-3 (2026-11-03), against 1e-4
(90th percentile) inside.

TWO LEVELS OF CELLS. The posterior is computed on a lattice `refine` times finer than the grid,
then again on a `zoom` x `zoom` lattice over the box holding its top `zoom_mass`; the zoomed
cells replace the coarse cells they cover. At sigma = 1e-3 the posterior is narrower than one
coarse cell, so without the zoom its credible regions would be a handful of cells.

WHAT A ROW SAYS. One row per (truth, K, sigma, prior beta), every row from the same noise draw
for a truth (its first K-1 entries):
  - `hpd_level`: the posterior mass of the cells whose density (relative to Kendall, so the
    region does not depend on a chart) exceeds the truth's. The truth lies in the
    alpha-credible region iff hpd_level <= alpha, so the fraction of rows with hpd_level <= alpha
    is the region's empirical coverage. It is exactly alpha on average when the truths are drawn
    from the prior used (`truth_betas` == prior beta); the other pairing measures misspecification.
  - `kendall90`, `kendall50`: the Kendall measure of the 90% / 50% credible region as a fraction
    of the support's -- how much of triangle space the data leave plausible.
  - `n90`: cells in the 90% region, a resolution check (flagged below 25).
  - `post_mean`, `map_ab`, `err_mean`: the posterior mean and mode in apex coordinates, and the
    mean's distance from the truth. `sigma_med`: the posterior median of sigma.

Map truths (`map_truths`, apexes snapped to the nearest interior grid point) also write their
cells to `<out>/maps/` for the posterior-map figure. Coverage truths are `n_truths` interior grid
points per entry of `truth_betas`, drawn without replacement with weights Kendall x
exp(-beta (E - E_eq)) (cell areas are equal), seeded by `seed`.

`prepare` writes `<out>/surrogate_<sha>.npz` (cross-validation, the coarse lattice, energies;
about 2 minutes, mostly 25k energies). `run` builds it if missing, so an array job can start
without it, but running `prepare` once first saves every task from building its own copy.
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
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.special import gammaln, logsumexp

from douse import spec as spec_mod

PACKAGE_ROOT = spec_mod.PACKAGE_ROOT
SCHEDULING = spec_mod.SCHEDULING
BLAS_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
N_FOLDS = 4
MIN_N90 = 25


# ---- spec -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PosteriorSpec:
    name: str
    grid: str                              # the family_grid spec, relative to the package root
    Ks: tuple = (3, 5, 10)
    sigmas: tuple = (1e-3, 1e-2)
    prior_betas: tuple = (0.0, 1.0)
    truth_betas: tuple = (0.0, 1.0)
    n_truths: int = 150
    map_truths: tuple = ((0.3, 0.5), (0.2, 0.25), (0.45, 0.75))
    seed: int = 0
    refine: int = 3
    zoom: int = 121
    zoom_mass: float = 0.9999
    ig_alpha: float = 1.0
    ig_beta: object = 'matched'    # a number, or 'matched': mode at the true sigma^2
    sigma_grid: tuple = (1e-5, 0.3, 61)    # log-spaced: lo, hi, count
    rbf_kernel: str = 'quintic'
    rbf_neighbors: int = 40
    rbf_degree: int = 2
    tasks: int = 20
    notes: str = ''
    slurm_mem: str | None = None
    slurm_time: str | None = None
    slurm_partition: str | None = None
    slurm_account: str | None = None
    source: str = field(default='', compare=False)

    def __post_init__(self):
        for k in ('Ks', 'sigmas', 'prior_betas', 'truth_betas', 'sigma_grid'):
            object.__setattr__(self, k, tuple(getattr(self, k)))
        object.__setattr__(self, 'map_truths', tuple(tuple(map(float, t)) for t in self.map_truths))
        if self.ig_beta != 'matched' and not isinstance(self.ig_beta, (int, float)):
            raise ValueError(f"ig_beta is a number or 'matched', got {self.ig_beta!r}")
        if min(self.Ks) < 2:
            raise ValueError('K >= 2: K eigenvalues give K - 1 ratios')

    def grid_spec(self):
        from benchmarks import family_grid as fg
        return fg.load(PACKAGE_ROOT/self.grid)

    def content(self):
        """What the rows depend on: everything but scheduling, notes, tasks and paths -- plus the
        grid spec's content hash, so a row cites the eigenvalues it was computed from."""
        skip = set(SCHEDULING) | {'source', 'notes', 'tasks'}
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in skip}
        d['grid'] = self.grid_spec().sha
        return d

    @property
    def sha(self):
        blob = json.dumps(self.content(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load(path):
    path = pathlib.Path(path)
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)
    known = set(PosteriorSpec.__dataclass_fields__) - {'source'}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f'{path}: unknown key(s) {unknown}; known: {sorted(known)}')
    raw.setdefault('name', path.stem)
    return PosteriorSpec(source=str(path), **raw)


def default_out(spec):
    return spec_mod.run_dir('triangle_posterior', spec.name)


# ---- the surrogate ----------------------------------------------------------------------------

def folds(P, h):
    """Lattice parity of each grid point: (i mod 2) + 2 (j mod 2), i, j its lattice indices."""
    I = np.rint(np.asarray(P)/h).astype(int)
    return (I[:, 0] % 2) + 2*(I[:, 1] % 2)


def interior(P, h):
    """Grid points with all four lattice neighbours in the grid."""
    I = np.rint(np.asarray(P)/h).astype(int)
    S = set(map(tuple, I))
    return np.array([all((i + di, j + dj) in S for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                     for i, j in I])


def in_support(a, b, min_angle_deg):
    """The grid's domain, vectorized: base [0, 1] longest, apex left, every angle >= the floor."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = a + 1j*b
    ang0, ang1 = np.degrees(np.angle(w)), np.degrees(np.angle(1 - w.conjugate()))
    return ((a >= 0) & (a <= 0.5) & (b > 0) & ((1 - a)**2 + b**2 <= 1 + 1e-12)
            & (np.minimum(np.minimum(ang0, ang1), 180 - ang0 - ang1) >= min_angle_deg - 1e-9))


def kendall(P):
    """Kendall's measure per unit area in apex coordinates, (centroid size)^-4 (normalized to 1
    at the equilateral apex 1/2 + i sqrt(3)/2)."""
    from douse.repulsive import kendall_apex_density
    w = np.asarray(P)[:, 0] + 1j*np.asarray(P)[:, 1]
    return kendall_apex_density(w) / kendall_apex_density(0.5 + 0.5j*np.sqrt(3))


def tri_energy(P):
    from douse.repulsive import energy
    return np.array([energy(np.array([0, 1, a + 1j*b])) for a, b in np.asarray(P)])


class Surrogate:
    """The interpolated forward map and everything the posterior reads that does not depend on a
    truth: cross-validation by fold, model error tau, the coarse lattice with its per-fold means,
    energies and Kendall weights."""

    def __init__(self, P, nu, h, min_angle_deg, refine, kernel='quintic', neighbors=40, degree=2,
                 energy_fn=tri_energy, E_grid=None, arrays=None):
        self.P, self.nu, self.h = np.asarray(P, float), np.asarray(nu, float), float(h)
        self.min_angle_deg, self.refine = float(min_angle_deg), int(refine)
        self.rbf_kw = dict(kernel=kernel, neighbors=neighbors, degree=degree)
        self.fold = folds(self.P, self.h)
        self.interior = interior(self.P, self.h)
        self.E_eq = float(energy_fn(np.array([[0.5, np.sqrt(3)/2]]))[0])
        self._rbf = {}
        if arrays is not None:
            for k, v in arrays.items():
                setattr(self, k, v)
        else:
            self._build(energy_fn, E_grid)
        self._E_rbf = RBFInterpolator(self.Pc, self.Ec, kernel='cubic', neighbors=20, degree=1)

    def rbf(self, f):
        """Fold f's interpolant: fitted to the other folds."""
        if f not in self._rbf:
            m = self.fold != f
            self._rbf[f] = RBFInterpolator(self.P[m], self.nu[m], **self.rbf_kw)
        return self._rbf[f]

    def _build(self, energy_fn, E_grid):
        self.cv_err = np.zeros_like(self.nu)
        for f in range(N_FOLDS):
            m = self.fold == f
            self.cv_err[m] = self.rbf(f)(self.P[m]) - self.nu[m]
        self.tau = np.sqrt((self.cv_err[self.interior]**2).mean(0))
        hc = self.h/self.refine
        a = hc*np.arange(int(round(0.5/hc)) + 1)
        b = hc*np.arange(1, int(np.ceil((np.sqrt(3)/2)/hc)) + 1)
        A, B = np.meshgrid(a, b)
        keep = in_support(A, B, self.min_angle_deg)
        self.hc = hc
        self.Pc = np.c_[A[keep], B[keep]]
        self.Mc = np.stack([self.rbf(f)(self.Pc) for f in range(N_FOLDS)])
        self.Ec = energy_fn(self.Pc)
        self.kc = kendall(self.Pc)
        self.E_grid = energy_fn(self.P) if E_grid is None else np.asarray(E_grid, float)

    ARRAYS = ('cv_err', 'tau', 'hc', 'Pc', 'Mc', 'Ec', 'kc', 'E_grid')

    def save(self, path, **meta):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
        np.savez(tmp, P=self.P, nu=self.nu, **{k: getattr(self, k) for k in self.ARRAYS},
                 meta=json.dumps(meta))
        os.replace(tmp, path)                  # atomic: concurrent tasks may race to write it

    @classmethod
    def load(cls, path, h, min_angle_deg, refine, **rbf_kw):
        z = np.load(path)
        arrays = {k: (float(z[k]) if z[k].ndim == 0 else z[k]) for k in cls.ARRAYS}
        return cls(z['P'], z['nu'], h, min_angle_deg, refine, arrays=arrays, **rbf_kw)

    def energy_at(self, Q):
        return self._E_rbf(Q)


def ig_beta(spec, sigma):
    return (spec.ig_alpha + 1)*sigma**2 if spec.ig_beta == 'matched' else float(spec.ig_beta)


def sigma_weights(spec, sigma):
    """`(sigma grid, log prior weight)`: sigma^2 ~ InvGamma(alpha, beta) on a log-sigma grid
    (the Jacobian of sigma^2 -> log sigma is sigma^2), normalized to sum to one. `sigma` is the
    true noise level, read only when `ig_beta` is 'matched'."""
    lo, hi, n = spec.sigma_grid
    s = np.geomspace(lo, hi, int(n))
    x = s**2
    al, be = spec.ig_alpha, ig_beta(spec, sigma)
    lw = al*np.log(be) - gammaln(al) - (al + 1)*np.log(x) - be/x + np.log(x)
    return s, lw - logsumexp(lw)


def log_lik(y, M, tau, sig, logw):
    """`(log marginal likelihood per cell (n,), joint log terms (n, s))`. y (k,), M (n, k), tau
    (k,), sig and logw (s,): y ~ N(M, (sigma^2 + tau^2) I) with sigma integrated over the grid."""
    v = sig[:, None]**2 + tau[None, :]**2                      # (s, k)
    r2 = (np.asarray(y)[None, :] - M)**2                       # (n, k)
    ll = -0.5*(r2 @ (1/v).T) - 0.5*np.log(2*np.pi*v).sum(1)[None, :] + logw[None, :]
    return logsumexp(ll, axis=1), ll


def hpd(logd, mass, logd_truth=None):
    """Credible regions on cells. `logd` the log density relative to the reference measure (Kendall
    here), `mass` the normalized posterior masses. Returns `(level per cell, level of the truth)`:
    a cell's level is the mass of every cell at least as dense, so the alpha-credible region is
    {level <= alpha}; the truth's is the mass strictly denser than it."""
    o = np.argsort(-logd, kind='stable')
    lev = np.empty_like(mass)
    lev[o] = np.cumsum(mass[o])
    t = None if logd_truth is None else float(mass[logd > logd_truth].sum())
    return lev, t


def _zoom_box(Pc, hc, lev, zoom_mass):
    sel = Pc[lev <= zoom_mass]
    if not len(sel):
        sel = Pc[np.argmin(lev)][None]
    lo = sel.min(0) - 1.5*hc
    hi = sel.max(0) + 1.5*hc
    return lo, hi


def posterior(sur, spec, f, y, K, beta, sigma, truth=None, E_truth=None, cells=False):
    """The posterior of one data vector `y` (K - 1 ratios) under fold `f`'s interpolant and the
    prior Kendall x exp(-beta (E - E_eq)). Returns a dict of summaries, and the cells when
    `cells` (for maps). `sigma` is the true noise level (for a matched sigma prior)."""
    k = K - 1
    sig, logw = sigma_weights(spec, sigma)
    tau = sur.tau[:k]

    def logpost(Q, M, E, kw, area):
        lm, ll = log_lik(y, M[:, :k], tau, sig, logw)
        lprior = -beta*(E - sur.E_eq)
        return lm + lprior, lm + lprior + np.log(kw*area), ll + (lprior + np.log(kw*area))[:, None]

    # coarse level
    ac = sur.hc**2
    ld_c, lm_c, _ = logpost(sur.Pc, sur.Mc[f], sur.Ec, sur.kc, ac)
    lev_c, _ = hpd(ld_c, np.exp(lm_c - logsumexp(lm_c)))
    lo, hi = _zoom_box(sur.Pc, sur.hc, lev_c, spec.zoom_mass)
    # zoom level: cell-centred lattice over the box, replacing the coarse cells inside it
    n = spec.zoom
    da, db = (hi - lo)/n
    ga = lo[0] + da*(np.arange(n) + 0.5)
    gb = lo[1] + db*(np.arange(n) + 0.5)
    A, B = np.meshgrid(ga, gb)
    keep = in_support(A, B, sur.min_angle_deg)
    Pz = np.c_[A[keep], B[keep]]
    outside = ~np.all((sur.Pc > lo) & (sur.Pc < hi), axis=1)
    Q = np.r_[sur.Pc[outside], Pz]
    area = np.r_[np.full(outside.sum(), ac), np.full(len(Pz), da*db)]
    M = np.r_[sur.Mc[f][outside], sur.rbf(f)(Pz)] if len(Pz) else sur.Mc[f][outside]
    E = np.r_[sur.Ec[outside], sur.energy_at(Pz)] if len(Pz) else sur.Ec[outside]
    kw = kendall(Q)
    ld, lm, ljoint = logpost(Q, M, E, kw, area)
    Z = logsumexp(lm)
    mass = np.exp(lm - Z)
    ld_t = None
    if truth is not None:
        lt, _ = log_lik(y, sur.rbf(f)(np.atleast_2d(truth))[:, :k], tau, sig, logw)
        ld_t = float(lt[0] - beta*(E_truth - sur.E_eq))
    lev, lev_t = hpd(ld, mass, ld_t)
    kmass = kw*area
    ktot = (sur.kc*ac).sum()
    ps = np.exp(logsumexp(ljoint, axis=0) - Z)                 # p(sigma | y) on the grid
    mean = mass @ Q
    out = dict(hpd_level=lev_t,
               kendall90=float(kmass[lev <= 0.9].sum()/ktot),
               kendall50=float(kmass[lev <= 0.5].sum()/ktot),
               n90=int((lev <= 0.9).sum()),
               post_mean=mean.tolist(), map_ab=Q[np.argmax(ld)].tolist(),
               sigma_med=float(sig[np.searchsorted(np.cumsum(ps), 0.5)]),
               zoom_box=[lo.tolist(), hi.tolist()], zoom_cells=int(len(Pz)))
    if truth is not None:
        out['err_mean'] = float(np.hypot(*(mean - np.asarray(truth))))
    if cells:
        out['cells'] = dict(Q=Q, mass=mass, level=lev, area=area)
    return out


# ---- the experiment ---------------------------------------------------------------------------

def surrogate_path(spec, out):
    return PACKAGE_ROOT/out/f'surrogate_{spec.sha}.npz' if not pathlib.Path(out).is_absolute() \
        else pathlib.Path(out)/f'surrogate_{spec.sha}.npz'


def grid_arrays(spec, grid_out=None):
    """`(P, lam, ids, grid spec)` from the collected triangle grid (suspect rows left out)."""
    from benchmarks import family_grid as fg
    g = spec.grid_spec()
    rows, rep = fg.collect(g, grid_out or fg.default_out(g))
    P, L, _, _, ids = fg.to_arrays(g, rows)
    if not len(P):
        raise FileNotFoundError(f'no rows for {g.name} under {grid_out or fg.default_out(g)}; '
                                f'pass --grid-out')
    if np.isnan(L[:, :max(spec.Ks)]).any():
        raise ValueError('the grid lacks eigenvalues up to max(Ks)')
    return P, L, ids, g


def get_surrogate(spec, out, grid_out=None, verbose=True):
    g = spec.grid_spec()
    path = surrogate_path(spec, out)
    kw = dict(kernel=spec.rbf_kernel, neighbors=spec.rbf_neighbors, degree=spec.rbf_degree)
    if path.exists():
        sur = Surrogate.load(path, g.h, g.min_angle_deg, spec.refine, **kw)
    else:
        P, L, ids, g = grid_arrays(spec, grid_out)
        t0 = time.time()
        sur = Surrogate(P, L[:, :1]/L[:, 1:], g.h, g.min_angle_deg, spec.refine, **kw)
        sur.save(path, spec_sha=spec.sha, ids=ids.tolist())
        if verbose:
            print(f'# built {path.name} in {time.time() - t0:.0f}s')
    z = np.load(path)
    sur.ids = np.array(json.loads(str(z['meta']))['ids'])
    return sur


def cv_report(sur, Ks):
    lines = ['# fold cross-validation of the interpolant, |interpolant - solved| per ratio set:']
    for K in Ks:
        e = np.abs(sur.cv_err[:, :K - 1]).max(1)
        lines.append(f'  K={K:2d}: interior median {np.median(e[sur.interior]):.1e}, '
                     f'90% {np.quantile(e[sur.interior], .9):.1e}, max {e[sur.interior].max():.1e}; '
                     f'edge max {e[~sur.interior].max():.1e}; tau_j <= {sur.tau[:K - 1].max():.1e}')
    return '\n'.join(lines)


def items(spec, sur):
    """The work list, a function of the spec: map truths first, then `n_truths` coverage truths
    per entry of `truth_betas`. Each item is `(key, kind, truth beta, grid row)`."""
    cand = np.flatnonzero(sur.interior)
    out = []
    for i, (a, b) in enumerate(spec.map_truths):
        r = cand[np.argmin(np.hypot(sur.P[cand, 0] - a, sur.P[cand, 1] - b))]
        out.append((f'map{i}', 'map', None, int(r)))
    rng = np.random.default_rng(spec.seed)
    kc = kendall(sur.P[cand])
    for tb in spec.truth_betas:
        w = kc*np.exp(-tb*(sur.E_grid[cand] - sur.E_eq))
        pick = rng.choice(cand, size=min(spec.n_truths, len(cand)), replace=False, p=w/w.sum())
        out += [(f'b{tb:g}_{j}', 'coverage', float(tb), int(r)) for j, r in enumerate(pick)]
    return out


def noise(spec, key):
    """The item's standard normal draw: one vector for every K, sigma and prior beta."""
    h = int(hashlib.sha256(f'{spec.seed}:{key}'.encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h).standard_normal(max(spec.Ks) - 1)


def run_item(spec, sur, item, out, prov):
    key, kind, tb, r = item
    truth = sur.P[r]
    f = int(sur.fold[r])
    eps = noise(spec, key)
    rows = []
    for K in spec.Ks:
        for s in spec.sigmas:
            y = sur.nu[r, :K - 1] + s*eps[:K - 1]
            for beta in spec.prior_betas:
                t0 = time.time()
                row = dict(key=key, kind=kind, truth_beta=tb, grid_id=int(sur.ids[r]),
                           a=float(truth[0]), b=float(truth[1]), fold=f, K=K, sigma=s,
                           prior_beta=beta)
                try:
                    res = posterior(sur, spec, f, y, K, beta, s, truth, sur.E_grid[r],
                                    cells=(kind == 'map'))
                    cells = res.pop('cells', None)
                    if cells is not None:
                        d = PACKAGE_ROOT/out/'maps' if not pathlib.Path(out).is_absolute() \
                            else pathlib.Path(out)/'maps'
                        d.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(d/f'{key}_K{K}_s{s:g}_b{beta:g}.npz', truth=truth,
                                            y=y, **cells)
                    row.update(res, status='ok')
                except Exception as exc:                             # noqa: BLE001
                    row.update(status='error', error=repr(exc), traceback=traceback.format_exc())
                row.update(wall=time.time() - t0, **prov)
                rows.append(row)
    return rows


def _done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)['key'])
    return done


def run(spec, task, ntasks, out, grid_out=None, limit=None):
    sur = get_surrogate(spec, out, grid_out)
    prov = spec_mod.provenance()
    prov.update(spec=spec.name, spec_sha=spec.sha, grid_sha=spec.grid_spec().sha)
    prov.pop('prec', None)
    prov.pop('backend', None)
    work = items(spec, sur)[task::ntasks]
    path = (PACKAGE_ROOT/out if not pathlib.Path(out).is_absolute() else pathlib.Path(out)) \
        / f'task_{task}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)
    todo = [it for it in work if it[0] not in done][:limit]
    print(f'# {spec.name} @ {spec.sha}: task {task}/{ntasks}, {len(work)} item(s), '
          f'{sum(it[0] in done for it in work)} done, running {len(todo)}', flush=True)
    for it in todo:
        t0 = time.time()
        rows = run_item(spec, sur, it, out, prov)
        with open(path, 'a') as fh:                  # one item's rows at once: resume by key
            for row in rows:
                fh.write(json.dumps(row) + '\n')
        bad = sum(r['status'] != 'ok' for r in rows)
        print(f'  {it[0]}: {len(rows)} posteriors in {time.time() - t0:.1f}s'
              + (f', {bad} error(s)' if bad else ''), flush=True)


# ---- collect / plot ---------------------------------------------------------------------------

def load_rows(spec, out):
    base = PACKAGE_ROOT/out if not pathlib.Path(out).is_absolute() else pathlib.Path(out)
    rows, foreign = [], 0
    for p in sorted(base.glob('task_*.jsonl')):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('spec_sha') != spec.sha:
                foreign += 1
                continue
            rows.append(r)
    return rows, foreign


def collect(spec, out, grid_out=None):
    rows, foreign = load_rows(spec, out)
    sur = get_surrogate(spec, out, grid_out, verbose=False)
    n_items = len(items(spec, sur))
    keys = {r['key'] for r in rows}
    err = [r for r in rows if r['status'] != 'ok']
    ok = [r for r in rows if r['status'] == 'ok']
    lines = [f'# {spec.name} @ {spec.sha}: {len(keys)}/{n_items} item(s), {len(rows)} row(s), '
             f'{len(err)} error(s), {foreign} row(s) from another spec_sha ignored',
             cv_report(sur, spec.Ks),
             f'  rows with fewer than {MIN_N90} cells in the 90% region: '
             f'{sum(r["n90"] < MIN_N90 for r in ok)}',
             '# coverage: fraction of truths inside the alpha-credible region (nominal 0.50/0.90/0.95)',
             f'  {"truths":>9s} {"prior":>6s} {"K":>3s} {"sigma":>7s} {"n":>4s}   '
             f'{"50%":>5s} {"90%":>5s} {"95%":>5s}   {"Kendall(90%)":>12s} {"|mean-truth|":>12s} '
             f'{"sigma_med":>9s}']
    cov = [r for r in ok if r['kind'] == 'coverage']
    for tb in spec.truth_betas:
        for beta in spec.prior_betas:
            for K in spec.Ks:
                for s in spec.sigmas:
                    rr = [r for r in cov if r['truth_beta'] == tb and r['prior_beta'] == beta
                          and r['K'] == K and r['sigma'] == s]
                    if not rr:
                        continue
                    h = np.array([r['hpd_level'] for r in rr])
                    lines.append(
                        f'  {"b=%g" % tb:>9s} {"b=%g" % beta:>6s} {K:3d} {s:7.0e} {len(rr):4d}   '
                        + ' '.join(f'{np.mean(h <= a):5.2f}' for a in (.5, .9, .95))
                        + f'   {np.median([r["kendall90"] for r in rr]):12.1e}'
                        f' {np.median([r["err_mean"] for r in rr]):12.1e}'
                        f' {np.median([r["sigma_med"] for r in rr]):9.1e}')
    if ok:
        w = np.array([r['wall'] for r in ok])
        lines.append(f'  wall per posterior: median {np.median(w):.2f}s, total {w.sum()/3600:.2f} CPU-h')
    return rows, '\n'.join(lines)


# ---- the equal-area chart --------------------------------------------------------------------
# Kendall's shape sphere for triangles, in Archimedes' cylindrical coordinates: longitude phi of
# zeta = (z0 + w z1 + w^2 z2)/(z0 + w^2 z1 + w z2), w = exp(2 pi i/3), folded into [0, 60] deg
# by relabelling and reflection, and height q = (1 - |zeta|^2)/(1 + |zeta|^2) = 4 sqrt(3) A /
# sum(l^2). The map is equal-area, so Kendall's measure is d(phi) dq: a density in this chart IS
# the density relative to Kendall. phi = 0 is isosceles with apex angle < 60 deg (the apex
# chart's left arc), phi = 60 isosceles with apex angle > 60 (a = 1/2), q = 1 equilateral.

W3 = np.exp(2j*np.pi/3)
_MINV = np.linalg.inv(np.array([[1, 1, 1], [1, W3, W3**2], [1, W3**2, W3]]))


def to_chart(a, b):
    """Apex a + ib (base [0, 1]) -> (phi in degrees, q)."""
    w = np.asarray(a) + 1j*np.asarray(b)
    zt = (W3 + W3**2*w)/(W3**2 + W3*w)
    zt = np.where(np.abs(zt) > 1, 1/np.conj(zt), zt)
    ph = np.degrees(np.angle(zt)) % 120
    ph = np.where(ph > 60, 120 - ph, ph)
    r2 = np.abs(zt)**2
    return ph, (1 - r2)/(1 + r2)


def from_chart(phi, q):
    """(phi in degrees, q) -> the canonical apex (a, b): longest side as base [0, 1], a <= 1/2."""
    phi, q = np.broadcast_arrays(np.asarray(phi, float), np.asarray(q, float))
    r = np.sqrt((1 - q)/(1 + q))
    rhs = np.stack([np.zeros_like(r), r*np.exp(1j*np.radians(phi)), np.ones_like(r)], -1)
    z = rhs @ _MINV.T                                          # (..., 3) vertices
    l = np.abs(np.roll(z, -1, axis=-1) - z)
    i = np.argmax(l, axis=-1)[..., None]
    p0 = np.take_along_axis(z, i, -1)[..., 0]
    p1 = np.take_along_axis(z, (i + 1) % 3, -1)[..., 0]
    p2 = np.take_along_axis(z, (i + 2) % 3, -1)[..., 0]
    w = (p2 - p0)/(p1 - p0)
    w = np.where(w.imag < 0, w.conj(), w)
    w = np.where(w.real > 0.5, 1 - w.conj(), w)
    return w.real, w.imag


def log_density(sur, spec, f, y, K, beta, sigma, Q):
    """Unnormalized log posterior density RELATIVE TO KENDALL at apex points Q (n, 2)."""
    sig, logw = sigma_weights(spec, sigma)
    lm, _ = log_lik(y, sur.rbf(f)(Q)[:, :K - 1], sur.tau[:K - 1], sig, logw)
    return lm - beta*(sur.energy_at(Q) - sur.E_eq)


def _chart_grid(lo, hi, n, min_angle_deg):
    """Cell-centred n x n lattice over the (phi, q) box [lo, hi]: `(PHI, QQ, apex points of the
    cells in the support, mask)`."""
    ph = lo[0] + (hi[0] - lo[0])*(np.arange(n) + .5)/n
    qq = lo[1] + (hi[1] - lo[1])*(np.arange(n) + .5)/n
    PHI, QQ = np.meshgrid(ph, qq)
    a, b = from_chart(PHI, QQ)
    m = in_support(a, b, min_angle_deg)
    return PHI, QQ, np.c_[a[m], b[m]], m


def support_area(min_angle_deg, n=800):
    """Area of the support in the (phi, q) chart, i.e. its Kendall measure in chart units."""
    _, _, P, m = _chart_grid((0, 0), (60, 1), n, min_angle_deg)
    return m.mean()*60.0


def plot_density(spec, out, sur, figdir, n=220):
    """Posterior densities in the equal-area chart: joint heatmaps (Kendall prior) and the
    marginals of phi and q (every prior), one pair of figures per sigma."""
    import matplotlib.pyplot as plt
    base = PACKAGE_ROOT/out if not pathlib.Path(out).is_absolute() else pathlib.Path(out)
    INK, MUTED = '#1f1f1e', '#8a897f'
    BETA_C = {b: c for b, c in zip(spec.prior_betas, ('#eb6834', '#2a78d6', '#1baf7a', '#7b5bd6'))}
    area = support_area(sur.min_angle_deg)
    mt = [it for it in items(spec, sur) if it[1] == 'map']

    def window(npz_paths, mass_level=0.999):
        pts = []
        for p in npz_paths:
            z = np.load(p)
            sel = z['Q'][z['level'] <= mass_level]
            pts.append(np.c_[to_chart(sel[:, 0], sel[:, 1])])
        pts = np.vstack(pts)
        lo, hi = pts.min(0), pts.max(0)
        pad = np.maximum((hi - lo)*0.15, (0.4, 0.004))
        return np.maximum(lo - pad, (0, 0)), np.minimum(hi + pad, (60, 1))

    def evaluate(key, r, K, s, beta, lo, hi):
        y = np.load(base/'maps'/f'{key}_K{K}_s{s:g}_b{beta:g}.npz')['y']
        PHI, QQ, P, m = _chart_grid(lo, hi, n, sur.min_angle_deg)
        ld = np.full(PHI.shape, -np.inf)
        ld[m] = log_density(sur, spec, int(sur.fold[r]), y, K, beta, s, P)
        cell = (hi[0] - lo[0])*(hi[1] - lo[1])/n**2
        mass = np.exp(ld - logsumexp(ld[m]))
        mass[~m] = 0.0
        return PHI, QQ, mass, cell

    for s in spec.sigmas:
        # joint densities, Kendall prior
        fig, axs = plt.subplots(len(mt), len(spec.Ks), figsize=(3.7*len(spec.Ks), 3.2*len(mt)),
                                squeeze=False)
        beta0 = spec.prior_betas[0]
        for i, (key, _, _, r) in enumerate(mt):
            tph, tq = to_chart(*sur.P[r])
            for j, K in enumerate(spec.Ks):
                ax = axs[i, j]
                lo, hi = window([base/'maps'/f'{key}_K{K}_s{s:g}_b{beta0:g}.npz'])
                PHI, QQ, mass, cell = evaluate(key, r, K, s, beta0, lo, hi)
                dens = mass/(cell/area)                       # relative to the Kendall prior
                im = ax.pcolormesh(PHI, QQ, np.ma.masked_where(mass == 0, dens), cmap='Blues',
                                   shading='nearest', rasterized=True)
                ax.contourf(PHI, QQ, (mass == 0).astype(float), levels=[0.5, 1.5],
                            colors=['#e8e7e3'])
                lev, _ = hpd(np.log(np.maximum(mass, 1e-300)).ravel(), mass.ravel())
                ax.contour(PHI, QQ, lev.reshape(mass.shape), levels=[0.5, 0.9], colors=INK,
                           linewidths=[1.1, .6])
                ax.plot(tph, tq, '+', color='#d95926', ms=11, mew=2)
                cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.02)
                cb.ax.tick_params(labelsize=6.5)
                cb.outline.set_visible(False)
                ax.set_title(f'apex ({sur.P[r, 0]:.2f}, {sur.P[r, 1]:.2f}), K={K}', fontsize=9)
                ax.tick_params(labelsize=7)
                if i == len(mt) - 1:
                    ax.set_xlabel('φ (degrees)', fontsize=8)
                if j == 0:
                    ax.set_ylabel('q (normalized area)', fontsize=8)
        fig.suptitle(f'Posterior density relative to Kendall, Kendall prior, σ = {s:g}, in the '
                     'equal-area chart\n(contours: 50% and 90% regions; + = truth; grey = outside '
                     'the grid; each panel zoomed)', fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir/f'density_sigma{s:g}.png', dpi=150, facecolor='white')
        plt.close(fig)

        # marginals of phi and q, every prior
        fig, axs = plt.subplots(len(mt), 2*len(spec.Ks), figsize=(2.6*2*len(spec.Ks), 2.4*len(mt)),
                                squeeze=False)
        for i, (key, _, _, r) in enumerate(mt):
            tph, tq = to_chart(*sur.P[r])
            for j, K in enumerate(spec.Ks):
                paths = [base/'maps'/f'{key}_K{K}_s{s:g}_b{b:g}.npz' for b in spec.prior_betas]
                lo, hi = window(paths)
                axp, axq = axs[i, 2*j], axs[i, 2*j + 1]
                for beta in spec.prior_betas:
                    PHI, QQ, mass, cell = evaluate(key, r, K, s, beta, lo, hi)
                    dph = (hi[0] - lo[0])/n
                    dq = (hi[1] - lo[1])/n
                    lab = 'Kendall' if beta == 0 else f'β={beta:g}'
                    axp.plot(PHI[0], mass.sum(0)/dph, color=BETA_C[beta], lw=1.3, label=lab)
                    axq.plot(QQ[:, 0], mass.sum(1)/dq, color=BETA_C[beta], lw=1.3, label=lab)
                for ax, t, name in ((axp, tph, 'φ (deg)'), (axq, tq, 'q')):
                    ax.axvline(t, color=INK, lw=.8, ls='--')
                    ax.set_ylim(0, None)
                    ax.tick_params(labelsize=6.5)
                    ax.set_yticks([])
                    if i == len(mt) - 1:
                        ax.set_xlabel(name, fontsize=8)
                axp.set_title(f'K={K}: φ', fontsize=8.5)
                axq.set_title(f'K={K}: q', fontsize=8.5)
                if j == 0:
                    axp.set_ylabel(f'apex ({sur.P[r, 0]:.2f}, {sur.P[r, 1]:.2f})', fontsize=8)
        axs[0, -1].legend(fontsize=7, frameon=False)
        fig.suptitle(f'Marginal posterior densities of φ and q, σ = {s:g} (dashed = truth; '
                     'each panel zoomed to the posteriors\' mass)', fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir/f'marginals_sigma{s:g}.png', dpi=150, facecolor='white')
        plt.close(fig)


def procrustes(w, t):
    """The triangle (0, 1, w) similarity-aligned to the triangle t (3 complex vertices, centred,
    unit centroid size), over the 6 labellings and reflection: returns the aligned vertices."""
    best = None
    for z in (np.array([0, 1, w]), np.array([0, 1, np.conj(w)])):
        for k in range(3):
            for zz in (np.roll(z, k), np.roll(z, k)[::-1]):
                zc = zz - zz.mean()
                fit = zc*(np.vdot(zc, t)/np.vdot(zc, zc))
                r = np.linalg.norm(fit - t)
                if best is None or r < best[0]:
                    best = (r, fit)
    return best[1]


def plot_samples(spec, out, sur, figdir, n_draw=80, n=220, seed=0):
    """Triangles drawn from the posterior (Kendall prior), each similarity-aligned to the truth
    (Procrustes over labellings and reflection), drawn translucent over the truth."""
    import matplotlib.pyplot as plt
    base = PACKAGE_ROOT/out if not pathlib.Path(out).is_absolute() else pathlib.Path(out)
    rng = np.random.default_rng(seed)
    mt = [it for it in items(spec, sur) if it[1] == 'map']
    beta = spec.prior_betas[0]
    for s in spec.sigmas:
        fig, axs = plt.subplots(len(mt), len(spec.Ks), figsize=(2.9*len(spec.Ks), 2.7*len(mt)),
                                squeeze=False)
        for i, (key, _, _, r) in enumerate(mt):
            t = np.array([0, 1, sur.P[r, 0] + 1j*sur.P[r, 1]])
            t = t - t.mean()
            t = t/np.linalg.norm(t)
            for j, K in enumerate(spec.Ks):
                ax = axs[i, j]
                y = np.load(base/'maps'/f'{key}_K{K}_s{s:g}_b{beta:g}.npz')['y']
                # sample on the full equal-area chart (cells are equal Kendall mass)
                PHI, QQ, P, m = _chart_grid((0, 0), (60, 1), n*2, sur.min_angle_deg)
                ld = log_density(sur, spec, int(sur.fold[r]), y, K, beta, s, P)
                # refine: resample on a zoom around the top of the coarse density
                wts = np.exp(ld - logsumexp(ld))
                ph_c, q_c = PHI[m], QQ[m]
                top = wts >= np.sort(wts)[::-1][np.searchsorted(np.cumsum(np.sort(wts)[::-1]),
                                                                0.9999)]
                lo = np.array([max(ph_c[top].min() - 60/(2*n), 0), max(q_c[top].min() - 1/(2*n), 0)])
                hi = np.array([min(ph_c[top].max() + 60/(2*n), 60), min(q_c[top].max() + 1/(2*n), 1)])
                PHI, QQ, P, m = _chart_grid(lo, hi, n, sur.min_angle_deg)
                ld = log_density(sur, spec, int(sur.fold[r]), y, K, beta, s, P)
                wts = np.exp(ld - logsumexp(ld))
                pick = rng.choice(len(P), n_draw, p=wts)
                dph, dq = (hi - lo)/n
                ph = PHI[m][pick] + dph*(rng.random(n_draw) - .5)
                qq = QQ[m][pick] + dq*(rng.random(n_draw) - .5)
                a, b = from_chart(ph, np.clip(qq, 0, 1))
                for w in a + 1j*b:
                    z = procrustes(w, t)
                    zc = np.r_[z, z[:1]]
                    ax.plot(zc.real, zc.imag, color='#2a78d6', lw=.6, alpha=.18)
                tc = np.r_[t, t[:1]]
                ax.plot(tc.real, tc.imag, color='#1f1f1e', lw=1.4, ls='--')
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title(f'apex ({sur.P[r, 0]:.2f}, {sur.P[r, 1]:.2f}), K={K}', fontsize=9)
        fig.suptitle(f'{n_draw} posterior draws (Kendall prior, σ = {s:g}), each aligned to the '
                     'truth (dashed) by rotation, scale, reflection and relabelling', fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir/f'samples_sigma{s:g}.png', dpi=150, facecolor='white')
        plt.close(fig)


def plot(spec, out, grid_out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    rows, rep = collect(spec, out, grid_out)
    print(rep)
    base = PACKAGE_ROOT/out if not pathlib.Path(out).is_absolute() else pathlib.Path(out)
    figdir = base/'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    sur = get_surrogate(spec, out, grid_out, verbose=False)
    ok = [r for r in rows if r['status'] == 'ok']
    INK, MUTED = '#1f1f1e', '#8a897f'
    BETA_C = {b: c for b, c in zip(spec.prior_betas, ('#eb6834', '#2a78d6', '#1baf7a', '#7b5bd6'))}

    # (1) posterior maps: one figure per sigma; rows = map truths, columns = K
    outline = _support_outline(sur)
    for s in spec.sigmas:
        mt = [it for it in items(spec, sur) if it[1] == 'map']
        fig, axs = plt.subplots(len(mt), len(spec.Ks), figsize=(3.6*len(spec.Ks), 3.0*len(mt)),
                                squeeze=False)
        for i, (key, _, _, r) in enumerate(mt):
            for j, K in enumerate(spec.Ks):
                ax = axs[i, j]
                ax.plot(*outline, color=MUTED, lw=.8)
                boxes = []
                for beta in spec.prior_betas:
                    p = base/'maps'/f'{key}_K{K}_s{s:g}_b{beta:g}.npz'
                    if not p.exists():
                        continue
                    z = np.load(p)
                    Q, lev = z['Q'], z['level']
                    tri = mtri.Triangulation(Q[:, 0], Q[:, 1])
                    ax.tricontour(tri, lev, levels=[0.5, 0.9], colors=BETA_C[beta],
                                  linewidths=[1.6, .9])
                    sel = Q[lev <= 0.99]
                    boxes.append((sel.min(0), sel.max(0)))
                ax.plot(*sur.P[r], '+', color=INK, ms=10, mew=1.5)
                if boxes:
                    lo = np.min([b[0] for b in boxes], 0)
                    hi = np.max([b[1] for b in boxes], 0)
                    c, w = (lo + hi)/2, max((hi - lo).max()*0.65, 0.01)
                    ax.set_xlim(c[0] - w, c[0] + w)
                    ax.set_ylim(c[1] - w, c[1] + w)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=7)
                ax.set_title(f'apex ({sur.P[r, 0]:.2f}, {sur.P[r, 1]:.2f}), K={K}', fontsize=9)
        handles = [plt.Line2D([], [], color=BETA_C[b], lw=1.6,
                              label='Kendall' if b == 0 else f'β={b:g}') for b in spec.prior_betas]
        fig.legend(handles=handles, loc='upper right', fontsize=9, frameon=False)
        fig.suptitle(f'Posterior 50% (thick) and 90% (thin) credible regions, σ = {s:g} '
                     f'(+ = truth; apex coordinates)', fontsize=11)
        fig.tight_layout()
        fig.savefig(figdir/f'maps_sigma{s:g}.png', dpi=150, facecolor='white')
        plt.close(fig)

    # (2) coverage: empirical vs nominal, one panel per (truths, prior)
    cov = [r for r in ok if r['kind'] == 'coverage']
    alphas = np.linspace(0, 1, 101)
    fig, axs = plt.subplots(len(spec.truth_betas), len(spec.prior_betas),
                            figsize=(3.4*len(spec.prior_betas), 3.3*len(spec.truth_betas)),
                            squeeze=False)
    for i, tb in enumerate(spec.truth_betas):
        for j, beta in enumerate(spec.prior_betas):
            ax = axs[i, j]
            ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls='--')
            for K, c in zip(spec.Ks, ('#86b6ef', '#3987e5', '#0d366b', '#000000')):
                for s, ls in zip(spec.sigmas, ('-', ':', '-.')):
                    h = np.array([r['hpd_level'] for r in cov if r['truth_beta'] == tb
                                  and r['prior_beta'] == beta and r['K'] == K and r['sigma'] == s])
                    if len(h):
                        ax.plot(alphas, (h[None, :] <= alphas[:, None]).mean(1), color=c, ls=ls,
                                lw=1.3, label=f'K={K}, σ={s:g}')
            ax.set_title(f'truths ~ {"Kendall" if tb == 0 else f"β={tb:g}"}, '
                         f'prior {"Kendall" if beta == 0 else f"β={beta:g}"}', fontsize=9)
            ax.set_xlabel('nominal level', fontsize=8)
            ax.set_ylabel('empirical coverage', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect('equal')
    axs[0, -1].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(figdir/'coverage.png', dpi=150, facecolor='white')
    plt.close(fig)

    # (3) identifiability: Kendall measure of the 90% region against K
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for beta in spec.prior_betas:
        for s, ls in zip(spec.sigmas, ('-', ':', '-.')):
            med = [np.median([r['kendall90'] for r in cov if r['truth_beta'] == beta
                              and r['prior_beta'] == beta and r['K'] == K and r['sigma'] == s]
                             or [np.nan]) for K in spec.Ks]
            ax.plot(spec.Ks, med, 'o', ls=ls, color=BETA_C[beta],
                    label=f'{"Kendall" if beta == 0 else f"β={beta:g}"}, σ={s:g}')
    ax.set_yscale('log')
    ax.set_xlabel('K (eigenvalues; K - 1 ratios)')
    ax.set_ylabel('Kendall measure of the 90% region\n(fraction of the support)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir/'identifiability.png', dpi=150, facecolor='white')
    plt.close(fig)

    # (4) the interpolant's cross-validated error over the grid
    fig, axs = plt.subplots(1, len(spec.Ks), figsize=(3.6*len(spec.Ks), 3.2), squeeze=False)
    from matplotlib.colors import LogNorm
    for ax, K in zip(axs[0], spec.Ks):
        e = np.abs(sur.cv_err[:, :K - 1]).max(1)
        sc = ax.scatter(sur.P[:, 0], sur.P[:, 1], c=e, s=4, cmap='viridis',
                        norm=LogNorm(1e-6, 1e-2))
        ax.plot(*outline, color=MUTED, lw=.8)
        ax.set_aspect('equal')
        ax.set_title(f'max_j<K |CV error|, K={K}', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=.8)
    fig.savefig(figdir/'cv_error.png', dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    plot_density(spec, out, sur, figdir)
    plot_samples(spec, out, sur, figdir)
    print(f'# wrote {figdir}')


def _support_outline(sur):
    """The support's boundary in apex coordinates, as (x, y) for plotting: the 10-degree angle
    curves and the circle (1 - a)^2 + b^2 = 1, traced as a fine contour."""
    a = np.linspace(0, 0.5, 201)
    b = np.linspace(1e-3, np.sqrt(3)/2, 301)
    A, B = np.meshgrid(a, b)
    keep = in_support(A, B, sur.min_angle_deg).astype(float)
    import matplotlib.pyplot as plt
    cs = plt.figure().add_subplot().contour(A, B, keep, levels=[0.5])
    seg = max(cs.allsegs[0], key=len)
    plt.close('all')
    return seg[:, 0], seg[:, 1]


# ---- slurm ------------------------------------------------------------------------------------

def sbatch_text(spec, spec_path, out, grid_out, mem, time_, partition, account, runner, mail=None):
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name=post_{spec.name}',
             f'#SBATCH --array=0-{spec.tasks - 1}',
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
    grid = f' --grid-out {shlex.quote(grid_out)}' if grid_out else ''
    lines += [f'#SBATCH --output={out}/logs/task_%a_%A.out',
              '',
              f'# spec {spec.name} @ {spec.sha} over {spec.tasks} task(s). Run `prepare` once first',
              '# (about 2 minutes) or every task builds the surrogate itself.',
              '# Generated by benchmarks/triangle_posterior.py; submit from the douse checkout.',
              'set -euo pipefail',
              'cd "${DOUSE_ROOT:-$SLURM_SUBMIT_DIR}"', '']
    lines += [f'export {v}=1' for v in BLAS_VARS]
    lines += ['',
              f'{runner} -m benchmarks.triangle_posterior run {shlex.quote(spec_path)} '
              f'--task ${{SLURM_ARRAY_TASK_ID}} --ntasks {spec.tasks} --out {shlex.quote(out)}'
              f'{grid}',
              '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'run', 'slurm', 'collect', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('spec')
        p.add_argument('--out', default=None,
                       help='output directory, relative to the douse checkout '
                            '(default $DOUSE_RUNS/triangle_posterior/<spec name>, else benchmarks/run/...)')
        p.add_argument('--grid-out', default=None,
                       help="the triangle grid's ledger directory (default family_grid's)")
        if name == 'run':
            p.add_argument('--task', type=int, default=0)
            p.add_argument('--ntasks', type=int, default=1)
            p.add_argument('--limit', type=int, default=None, help='run at most this many items')
        if name == 'slurm':
            p.add_argument('--mem', default=None)
            p.add_argument('--time', default=None)
            p.add_argument('--partition', default=None)
            p.add_argument('--account', default=None)
            p.add_argument('--mail', default=None)
            p.add_argument('--runner', default='uv run python')
            p.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    spec = load(args.spec)
    out = args.out or default_out(spec)

    if args.cmd == 'prepare':
        sur = get_surrogate(spec, out, args.grid_out)
        print(f'# {spec.name} @ {spec.sha}: {len(sur.P)} grid points ({sur.interior.sum()} '
              f'interior), {len(sur.Pc)} coarse cells (h/{spec.refine}), '
              f'{len(items(spec, sur))} item(s) x {len(spec.Ks)*len(spec.sigmas)*len(spec.prior_betas)} '
              f'posteriors')
        print(cv_report(sur, spec.Ks))
        return 0
    if args.cmd == 'run':
        run(spec, args.task, args.ntasks, out, args.grid_out, args.limit)
        return 0
    if args.cmd == 'collect':
        print(collect(spec, out, args.grid_out)[1])
        return 0
    if args.cmd == 'plot':
        plot(spec, out, args.grid_out)
        return 0

    from benchmarks.slurm import mail_address
    spec_path = str(pathlib.Path(args.spec).resolve().relative_to(PACKAGE_ROOT)) \
        if pathlib.Path(args.spec).resolve().is_relative_to(PACKAGE_ROOT) else args.spec
    text = sbatch_text(spec, spec_path, out, args.grid_out, args.mem or spec.slurm_mem or '4G',
                       args.time or spec.slurm_time or '4:00:00',
                       args.partition or spec.slurm_partition, args.account or spec.slurm_account,
                       args.runner, mail_address(args.mail))
    path = pathlib.Path(out)/f'{spec.name}.sbatch'
    if args.dry_run:
        print(text)
        return 0
    dest = path if path.is_absolute() else PACKAGE_ROOT/path
    (dest.parent/'logs').mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f'# wrote {path}. From the douse checkout, build the surrogate once, then submit:\n'
          f'uv run python -m benchmarks.triangle_posterior prepare {spec_path}\n'
          f'sbatch {path}\n'
          f'# then: uv run python -m benchmarks.triangle_posterior plot {spec_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
