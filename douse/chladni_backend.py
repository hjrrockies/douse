"""douse's backend: `SpectrumEvaluator` on `chladni`.  (the default since 2026-10-14)

`chladni` (`../chladni`, the rebuild of `lappy`) plans its own basis per shape family, solves to a
requested `prec` with a certified bound and flags per level, and supplies the Hadamard matrices
(`chladni.eigensolver.Spectrum.hadamard`). This module adapts that to `douse`'s evaluator
interface. The lappy backend survives, deprecated, as `douse.evaluate.LappySpectrumEvaluator`
(`douse[lappy]`); `ChladniEvaluator` is this class's name from before the switch.

    ev = SpectrumEvaluator(VertexPolygon(5), n_eigs=3, prec=1e-10)
    lams, J = ev(p)                       # the spectrum and its (K, n_p) Jacobian
    lams, solver = ev.eigenvalues(p)      # `solver` works with `douse.derivative`

**What it relies on, and so does not re-implement.** `lappy`'s backend wraps the solve in guards
douse grew from lost runs: spurious values, the Weyl deficit with escalation, `replan_tol`,
tracking. `chladni`'s `Spectrum.flags` reports the same failures from inside the solver:
- a level that misses `prec` (its certified bound, Moler-Payne cross-checked);
- the Weyl count audit (`short` / `excess`);
- an uncertified count;
- a decline;
- a planner that found no basis meeting `prec`.

`refusal_for` maps any of those onto `douse`'s `ShapeRefusal` classes, so a caller that shrinks
on `ShapeRefusal` keeps working unchanged.

**Plan once, realize per iterate.** The `chladni.plan.Plan` is built on the first shape (or
`plan_at`) and realized on every later one. A shape with a different corner count (a vertex gone
flat) cannot realize it; that triggers one re-plan at that shape, counted in `n_replans`.

**Precision.** `prec` defaults to 1e-10, `chladni`'s first working precision for douse
(`chladni/docs/design/douse_requirements.md`). `lappy`'s backend defaults to 1e-13.

**Gradients.** `douse.derivative.dlams_dp` and `cluster_matrices` dispatch to the handle this
returns, which evaluates the vertex velocity fields as `V(curve, t)` on `chladni`'s own Hadamard
rule. A multiple level refuses the simple-eigenvalue gradient (`DegenerateSpectrum`), as the
`lappy` path's policy does; `cluster_matrices` is the path there.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np

import time

from .backend import RunHealth, SolveTrace
from .errors import DegenerateSpectrum, SolverRefusal, UnresolvedSpectrum

__all__ = ["SpectrumEvaluator", "ChladniEvaluator", "ChladniSolver", "refusal_for"]


@contextmanager
def _quiet():
    """Silence ONE warning: Fourier-Bessel's per-column accuracy bound ("FourierBessel can
    deliver about 1e-9 ..."). Measured against 30-digit mpmath it is ~1e5x pessimistic
    (`chladni/notes/hard_cases.md`, part 3), so in a long run it is noise. Every other
    `chladni` warning still reaches the caller; shortfalls arrive as flags and refusals."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FourierBessel can deliver")
        yield


def refusal_for(spec):
    """The `douse` exception a flagged `chladni` spectrum should raise, or None.

    `count audit: unaudited` is not a failure (the audit could not be applied), so it does not
    refuse. Every other flag means the returned set cannot be trusted to be lam_1 .. lam_K at
    `prec`.
    """
    if getattr(spec, "declined", False):
        return SolverRefusal(f"chladni declined this shape: {'; '.join(spec.flags)}")
    bad = [f for f in spec.flags if not f.startswith("count audit: unaudited")]
    if bad:
        return UnresolvedSpectrum("chladni: " + "; ".join(bad))
    return None


def _shoelace(z):
    return 0.5 * float(np.sum(z.real * np.roll(z.imag, -1) - np.roll(z.real, -1) * z.imag))


class ChladniSolver:
    """The handle `ChladniEvaluator.eigenvalues` returns: one solved shape, with what
    `douse.derivative` needs from it."""

    def __init__(self, record, vertices, degenerate_tol=1e-6):
        self.record = record                      # chladni's SolveRecord: plan, solver, ...
        self.spec = record.spectrum               # the value: levels with multiplicities
        self.vertices = np.asarray(vertices, dtype=complex)
        self.degenerate_tol = float(degenerate_tol)
        self._H = {}

    @property
    def spectrum(self):
        """The `chladni.eigensolver.Spectrum` of this shape (`SolvedShape`)."""
        return self.spec

    @property
    def basis(self):
        """The realized basis this shape was solved on (its size is the plan's, on every shape)."""
        return self.record.basis

    def _velocity(self):
        """`V(curve, t)` for the 2N vertex-coordinate fields, `[x_0..x_{N-1}, y_0..y_{N-1}]`:
        moving vertex k displaces its two edges linearly in t, and V is that displacement's normal
        component (outward normal, `chladni` boundaries run counter-clockwise)."""
        dom = self.record.domain
        segs = [s for b in dom.boundaries for s in b.segments]
        verts = self.vertices
        n = len(verts)
        ends = [(int(np.argmin(np.abs(verts - s.p0))), int(np.argmin(np.abs(verts - s.p1))))
                for s in segs]
        normals = [-1j * (s.p1 - s.p0) / abs(s.p1 - s.p0) for s in segs]

        def V(curve, t):
            t = np.asarray(t, dtype=float)
            i0, i1 = ends[curve]
            nrm = normals[curve]
            out = np.zeros((2 * n, len(t)))
            for axis, d in ((0, 1.0 + 0j), (1, 1j)):
                vn = d.real * nrm.real + d.imag * nrm.imag      # (d . n) per unit displacement
                out[axis * n + i0] += (1.0 - t) * vn
                out[axis * n + i1] += t * vn
            return out
        return V

    def _level(self, lam):
        return self.spec.level_of(lam)

    def hadamard_vertices(self, i):
        """`(2N, m, m)` for level `i`, cached."""
        if i not in self._H:
            with _quiet():
                self._H[i] = self.record.hadamard(i, self._velocity())
        return self._H[i]

    def dlams_dvertices(self, lams, n_vert):
        """`(K, 2N)`: the simple-eigenvalue gradients (`SolvedShape`)."""
        J = np.empty((len(lams), 2 * n_vert))
        for j, lam in enumerate(lams):
            i = self._level(lam)
            if self.spec.mult[i] > 1:
                raise DegenerateSpectrum(
                    f"lam = {lam:.10g} is a level of multiplicity {self.spec.mult[i]}: its shape "
                    f"derivative is the spectrum of an m x m matrix, not a gradient. Use "
                    f"douse.derivative.cluster_matrices.")
            J[j] = self.hadamard_vertices(i)[:, 0, 0]
        return J

    def _cluster_levels(self, lam, mult, lams):
        """The level indices making up a cluster of `mult` eigenvalues near `lam`
        (`Spectrum.cluster`): those of the members `lams` when given, else the adjacent levels
        whose multiplicities sum to `mult`."""
        idx, total = self.spec.cluster(lams, near=lam, mult=mult)
        if total != int(mult):
            # A refusal, not a bug: the caller's eps-active cluster and the solver's levels
            # disagree on how many eigenvalues sit here, so no block is honest. Raised as a
            # `ShapeRefusal` so an optimizer rejects the trial and shrinks rather than dying.
            raise UnresolvedSpectrum(
                f"chladni: the cluster of {mult} eigenvalues near lam = {lam:.10g} does not match "
                f"the solver's levels: " + ", ".join(
                    f"{float(self.spec.lam[i]):.10g} x{int(self.spec.mult[i])}" for i in idx)
                + f" hold {sum(int(self.spec.mult[i]) for i in idx)}")
        return idx

    def cluster_matrices_dvertices(self, family, p, lam, mult, lams=None):
        """`(n_p, m, m)` for the cluster of `mult` eigenvalues near `lam` (`SolvedShape`). One
        level of multiplicity `m`, or several distinct levels -- a nearly degenerate cluster --
        taken as ONE block with their cross terms (`SolveRecord.hadamard([i, j, ...])`)."""
        idx = self._cluster_levels(lam, mult, lams)
        key = tuple(idx)
        if key not in self._H:
            with _quiet():
                self._H[key] = self.record.hadamard(idx[0] if len(idx) == 1 else idx,
                                                    self._velocity())
        H = self._H[key]                                        # (2N, m, m)
        Jv = family.d_vertices_dp(p)                            # (2N, n_p)
        return np.einsum("fij,fp->pij", H, Jv)

    # The names `douse.derivative` dispatched on before the `SolvedShape` protocol (2026-10-14).
    chladni_dlams_dvertices = dlams_dvertices
    chladni_cluster_matrices = cluster_matrices_dvertices


class SpectrumEvaluator:
    """`SpectrumEvaluator`'s interface on the `chladni` backend. See the module docstring."""

    #: lappy-backend options accepted for compatibility and ignored here: chladni's solve has
    #: its own precision checks, tracking and re-planning (2026-10-14, when chladni became the
    #: default backend). Each warns once.
    LAPPY_ONLY = ("target", "lam_max_window", "track_ttol", "use_track_set",
                  "replan_tol", "enforce_weyl", "spurious_tol")

    def __init__(self, family, n_eigs=1, prec=1e-10, plan_at=None, degenerate_tol=1e-6,
                 confirm=False, warn_gap=1e-3, **lappy_options):
        unknown = sorted(set(lappy_options) - set(self.LAPPY_ONLY))
        if unknown:
            raise TypeError(f"unexpected option(s) {unknown}")
        for name in sorted(lappy_options):
            _warn_once(name)
        self.family = family
        self.n_eigs = int(n_eigs)
        self.prec = float(prec)
        self.degenerate_tol = float(degenerate_tol)
        self.warn_gap = float(warn_gap)
        self.min_gap = np.inf               # smallest relative gap seen, over every evaluation
        self.confirm = bool(confirm)
        self._plan = None
        self._plan_at = None if plan_at is None else np.asarray(plan_at, dtype=float)
        self._plan_built_at = None
        self._last = None                   # (p, lams): the continuation seed, as cluster_opt sets it
        self._specs = {}                    # p.tobytes() -> Spectrum, the last two shapes solved
        self._last_spec = None
        self.last_sigma = np.nan            # worst attained relative bound at the last shape
        # RUN HEALTH, with chladni's meaning (2026-10-14). The ledger (`benchmarks/
        # ratio_multistart.py`, `summarize_grid.py`) filters rows on these names, which the lappy
        # evaluator coined for its tension statistics. Here they are read off each accepted
        # `Spectrum`, and are BOUNDS, not tensions:
        #   worst_sigma        running max of `last_sigma`, the worst certified relative bound
        #                      over the returned levels. On a healthy run it sits near `prec`;
        #                      the ledger's `sigma_max = 1e-6` gate is then 4 orders above it.
        #   worst_sigma_any    the same number: every returned level is certified, so one
        #                      statistic serves where lappy needed two.
        #   worst_weyl_deficit running max of the count audit's mean Weyl deficit
        #                      (`chladni.asymp.CountAudit.statistic`) where it was computed.
        #   n_weyl_flags       audits that did not read "ok" (including "unaudited").
        #   n_weyl_refusals, n_spurious_refusals: `UnresolvedSpectrum`s by cause -- the count
        #                      audit, or a level unmet/uncertain/short.
        self.worst_sigma = -np.inf
        self.worst_sigma_any = -np.inf
        self.worst_weyl_deficit = -np.inf
        self.n_weyl_flags = 0
        self.n_weyl_refusals = 0
        self.n_spurious_refusals = 0
        #: Tension evaluations summed over every accepted solve (`Spectrum.diagnostics`): what
        #: the run cost in the solver's own currency, for the ledger.
        self.n_tension_evals = 0
        self.n_solves = 0                   # cold solves
        self.n_tracks = 0                   # warm solves accepted
        self.n_track_refusals = 0           # warm solves asked for and refused (then solved cold)
        self.last_warm_note = None
        self.n_replans = 0
        self.n_solver_refusals = 0
        self.refusals = []
        self.max_refusals_kept = 16
        #: One `SolveTrace` per solve -- cold / warm / replan, why, rung, evaluations, seconds --
        #: capped, for the ledger and for diagnosing a slow run (`health().trace`).
        self.trace = []
        self.max_trace_kept = 400

    def __repr__(self):
        return (f"SpectrumEvaluator({self.family!r}, n_eigs={self.n_eigs}, "
                f"prec={self.prec:.0e})")

    def _domain(self, p):
        from chladni.geometry import Polygon
        v = np.asarray(self.family.vertices(p), dtype=complex)
        if _shoelace(v) < 0:
            raise ValueError("this p gives a clockwise polygon, which is outside the "
                             "parametrization's range")
        return Polygon(v, bc="dir"), v

    def plan(self, p):
        """The frozen `chladni.plan.Plan`, built on first use at `plan_at` (default: first `p`)."""
        if self._plan is None:
            from chladni.eigenproblem import Eigenproblem
            at = self._plan_at if self._plan_at is not None else np.asarray(p, dtype=float)
            self._plan_built_at = np.array(at, dtype=float)
            dom, _ = self._domain(at)
            with _quiet():
                self._plan = Eigenproblem(dom).plan(self.n_eigs, prec=self.prec,
                                                    confirm=self.confirm)
        return self._plan

    def eigenvalues(self, p, seed=None):
        """`(lams, solver)`; raises `ShapeRefusal` (and nothing else) when the shape is the
        problem, as `SpectrumEvaluator.eigenvalues` does.

        **Plan drift is re-planned, not refused.** A plan frozen at an earlier shape can stop
        meeting `prec` as the shape moves: a level goes unmet, the audit reads short or excess,
        or the search declines with a basis sized for the old shape's wavenumbers. That is
        routine over an optimization run (`lappy`'s backend re-plans on it through `replan_tol`).
        So any refusal on a plan that was NOT made at this shape triggers one re-plan here and a
        second solve, and only a refusal on a plan made AT this shape is passed on (2026-10-13:
        without this, 4 of the first 5 comparison runs stopped short of the optimum).

        **Tracked from the previous shape** when `_last` names one solved here: chladni's warm
        start (`Eigenproblem.solve(..., near=prev)`), which checks continuity and solves cold on
        any failure, and every 10 warm solves. `seed` is accepted for interface parity and not
        used: a warm start needs the previous SPECTRUM (its plan and levels), not just values."""
        p = np.asarray(p, dtype=float)
        dom, verts = self._domain(p)
        spec, exc = self._solve(dom, self.plan(p), p, near=self._near())
        if exc is not None and not self._planned_here(p):
            self.replan(p)
            self.n_replans += 1
            spec, exc = self._solve(dom, self.plan(p), p)
            if self.trace:
                self.trace[-1] = SolveTrace(kind="replan", reason=self.trace[-1].reason,
                                            rung=self.trace[-1].rung,
                                            n_evals=self.trace[-1].n_evals,
                                            seconds=self.trace[-1].seconds)
        if exc is not None:
            self._record(p, str(exc), spec)
            if isinstance(exc, UnresolvedSpectrum):
                if "count audit" in str(exc):
                    self.n_weyl_refusals += 1
                else:
                    self.n_spurious_refusals += 1
            raise exc
        with np.errstate(invalid="ignore", divide="ignore"):
            self.last_sigma = float(np.nanmax(spec.bound / spec.lam)) if spec.lam.size else np.nan
        self._health(spec)
        # THE WHOLE TOP LEVEL, never cut. `Spectrum` returns the level at `n_eigs` whole, so
        # `len(lams) >= n_eigs`, with equality unless a multiple level straddles the request. A
        # truncated copy is a cluster `maximize_ratio` would count as `m` members while the
        # solver holds `m + 1` -- a mismatch `ChladniSolver._cluster_levels` must refuse -- and
        # on `lam_3/lam_1` over hexagons, where the run drives lam_3 into the lam_4 = lam_5
        # pair, that was the standard problem's own geometry (2026-10-14). Consumers index by
        # position (`lams[k]`); `ObjectiveFunction` slices to its objective's count.
        lams = spec.unfolded
        self._check_gap(lams[:self.n_eigs])     # before the seed moves, as on the lappy backend
        self._remember(p, spec, lams)
        return lams, ChladniSolver(spec, verts, self.degenerate_tol)

    def reseed(self, p, lams, solved=None):
        """Point warm tracking at the accepted iterate `(p, lams)` after a rejected trial
        (`SpectrumBackend`). With `solved` (the `ChladniSolver` returned for that iterate) its
        record is put back so the next warm solve continues from it even after two trial
        solves evicted it -- which used to force a cold solve."""
        p = np.asarray(p, dtype=float)
        self._last = (p.copy(), np.array(lams, dtype=float))
        rec = getattr(solved, "record", None)
        if rec is not None:
            self._specs.pop(p.tobytes(), None)
            self._specs[p.tobytes()] = rec
            while len(self._specs) > 2:
                self._specs.pop(next(iter(self._specs)))

    def health(self):
        """The run's statistics as a value (`RunHealth`), with chladni's meaning."""
        return RunHealth(worst_bound=self.worst_sigma, last_bound=self.last_sigma,
                         worst_deficit=self.worst_weyl_deficit, n_replans=self.n_replans,
                         n_solves=self.n_solves, n_warm=self.n_tracks,
                         n_warm_refused=self.n_track_refusals,
                         n_tension_evals=self.n_tension_evals, n_audit_flags=self.n_weyl_flags,
                         n_audit_refusals=self.n_weyl_refusals,
                         n_unmet_refusals=self.n_spurious_refusals,
                         n_solver_refusals=self.n_solver_refusals, min_gap=self.min_gap,
                         refusals=tuple(self.refusals), plan_at=self._plan_at,
                         plan_built_at=self._plan_built_at, trace=tuple(self.trace))

    def _health(self, spec):
        """Fold an accepted spectrum into the run statistics (see `__init__`)."""
        if np.isfinite(self.last_sigma):
            self.worst_sigma = max(self.worst_sigma, self.last_sigma)
            self.worst_sigma_any = self.worst_sigma
        d = getattr(spec, "diagnostics", None)
        if d is not None:
            self.n_tension_evals += int(getattr(d, "evaluations", 0) or 0)
        audit = getattr(spec, "count_audit", None)
        if audit is not None:
            if audit.verdict != "ok":
                self.n_weyl_flags += 1
            if np.isfinite(audit.statistic):
                self.worst_weyl_deficit = max(self.worst_weyl_deficit, float(audit.statistic))

    def _check_gap(self, lams):
        """The degeneracy guard, as the lappy evaluator has it: raise below `degenerate_tol`,
        where an index-based gradient is a different quantity (`DegenerateSpectrum`), and warn
        below `warn_gap`. Pass `degenerate_tol=0` to run through crossings on cluster objectives."""
        if len(lams) < 2:
            return
        s = np.sort(np.asarray(lams, dtype=float))
        gap = float(np.min(np.diff(s) / s[:-1]))
        self.min_gap = min(self.min_gap, gap)
        if gap < self.degenerate_tol:
            raise DegenerateSpectrum(
                f'two of the tracked eigenvalues are within {gap:.2e} (relative) at this shape, '
                f'below degenerate_tol={self.degenerate_tol:.1e}. The simple-eigenvalue gradient '
                f'is not an approximation here, it is a different quantity -- see '
                f'DegenerateSpectrum. Use the m x m cluster matrix, or an objective that is a '
                f'symmetric function of the cluster, or raise degenerate_tol if you know this '
                f'crossing is benign.')
        if gap < self.warn_gap:
            warnings.warn(
                f'the tracked eigenvalues are within {gap:.2e} (relative) of each other at this '
                f'shape. An index-based objective is not differentiable through a crossing, and '
                f'these problems tend to optimize TOWARDS one, so a gradient method may stall or '
                f'chatter here rather than converge.')

    def _near(self):
        """The Spectrum behind the continuation seed `_last = (p, lams)`, or None."""
        if self._last is None:
            return None
        return self._specs.get(np.asarray(self._last[0], dtype=float).tobytes())

    def _remember(self, p, spec, lams):
        """Keep this shape's spectrum (and the one before it: the accepted point, when this was
        a trial) and make it the seed."""
        key = p.tobytes()
        self._specs.pop(key, None)
        self._specs[key] = spec
        while len(self._specs) > 2:
            self._specs.pop(next(iter(self._specs)))
        self._last_spec = spec
        self._last = (p.copy(), np.array(lams, dtype=float))

    def _planned_here(self, p):
        return (self._plan_built_at is not None
                and np.array_equal(self._plan_built_at, np.asarray(p, dtype=float)))

    def _solve(self, dom, plan, p, near=None):
        """`(spec, refusal_or_None)` for `dom` on `plan`, warm from `near` when it was solved on
        the same plan. A plan that cannot be realized here (a different corner count) is
        re-planned at `p` immediately."""
        from chladni.eigenproblem import Eigenproblem
        if near is not None and near.plan is not plan:
            near = None
        t0, kind = time.perf_counter(), "cold"
        try:
            try:
                with _quiet():
                    spec = Eigenproblem(dom).solve(self.n_eigs, prec=self.prec, plan=plan,
                                                   near=near)
            except ValueError as e:
                if "corner" not in str(e):
                    raise
                self.replan(p)                  # a different corner structure: re-plan here
                self.n_replans += 1
                kind = "replan"
                with _quiet():
                    spec = Eigenproblem(dom).solve(self.n_eigs, prec=self.prec,
                                                   plan=self.plan(p))
        except (RuntimeError, ValueError) as e:
            self.n_solver_refusals += 1
            self._trace(kind, f"{type(e).__name__}: {e}", plan, None, t0)
            exc = SolverRefusal(f"chladni could not solve this shape: {e}")
            exc.__cause__ = e
            return None, exc
        if spec.warm:
            self.n_tracks += 1
            kind = "warm"
        else:
            self.n_solves += 1
            if near is not None:
                self.n_track_refusals += 1
        self.last_warm_note = spec.warm_note
        refusal = refusal_for(spec)
        self._trace(kind, spec.warm_note if refusal is None else str(refusal), spec.plan,
                    spec, t0)
        return spec, refusal

    def _trace(self, kind, reason, plan, spec, t0):
        if len(self.trace) < self.max_trace_kept:
            d = None if spec is None else getattr(spec, "diagnostics", None)
            self.trace.append(SolveTrace(
                kind=kind, reason=reason, rung=getattr(plan, "rung", None),
                n_evals=int(getattr(d, "evaluations", 0) or 0),
                seconds=time.perf_counter() - t0))

    def _record(self, p, error, spec=None):
        if len(self.refusals) < self.max_refusals_kept:
            self.refusals.append(dict(
                p=[float(t) for t in np.asarray(p, dtype=float)],
                plan_at=(None if self._plan_built_at is None
                         else [float(t) for t in self._plan_built_at]),
                lams=(None if spec is None else [float(t) for t in spec.unfolded]),
                flags=(None if spec is None else list(spec.flags)),
                n_replans=int(self.n_replans), error=error))

    def __call__(self, p, seed=None):
        """`(lams, dlams_dp)` -- the spectrum and its `(K, n_p)` Jacobian."""
        from .derivative import dlams_dp
        lams, solver = self.eigenvalues(p, seed=seed)
        return lams, dlams_dp(self.family, p, solver, lams)

    def replan(self, p=None):
        """Throw the frozen plan away so the next call rebuilds it (at `p` if given)."""
        self._plan = None
        self._plan_at = None if p is None else np.asarray(p, dtype=float)
        self.reset_continuation()

    def reset_continuation(self):
        """Forget the previous shape, so the next call solves cold. Use after a jump."""
        self._last = None
        self._specs = {}


#: The name this evaluator had while lappy was douse's default (2026-10-12 .. 10-14).
ChladniEvaluator = SpectrumEvaluator

_WARNED = set()


def _warn_once(name):
    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(f"SpectrumEvaluator ignores the lappy-backend option {name!r}: chladni's "
                      f"solve has its own precision checks, tracking and re-planning",
                      DeprecationWarning, stacklevel=3)
