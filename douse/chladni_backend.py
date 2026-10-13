"""The `chladni` backend: `ChladniEvaluator`, a drop-in for `SpectrumEvaluator`.

`chladni` (`../chladni`, the rebuild of `lappy`) plans its own basis per shape family, solves to a
requested `prec` with a certified bound and flags per level, and supplies the Hadamard matrices
(`chladni.eigensolver.Spectrum.hadamard`). This module adapts that to `douse`'s evaluator
interface, so an optimizer can switch backends by constructing one evaluator instead of the
other:

    ev = ChladniEvaluator(VertexPolygon(5), n_eigs=3, prec=1e-10)
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

from .evaluate import DegenerateSpectrum, SolverRefusal, UnresolvedSpectrum

__all__ = ["ChladniEvaluator", "ChladniSolver", "refusal_for"]


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

    def __init__(self, spec, vertices, degenerate_tol=1e-6):
        self.spec = spec
        self.vertices = np.asarray(vertices, dtype=complex)
        self.degenerate_tol = float(degenerate_tol)
        self._H = {}

    def _velocity(self):
        """`V(curve, t)` for the 2N vertex-coordinate fields, `[x_0..x_{N-1}, y_0..y_{N-1}]`:
        moving vertex k displaces its two edges linearly in t, and V is that displacement's normal
        component (outward normal, `chladni` boundaries run counter-clockwise)."""
        dom = self.spec.source.domain
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
        i = int(np.argmin(np.abs(self.spec.lam - float(lam))))
        if abs(self.spec.lam[i] - lam) > 1e-6 * abs(lam):
            raise ValueError(f"lam = {lam} is not a level of this spectrum")
        return i

    def hadamard_vertices(self, i):
        """`(2N, m, m)` for level `i`, cached."""
        if i not in self._H:
            with _quiet():
                self._H[i] = self.spec.hadamard(i, self._velocity())
        return self._H[i]

    def chladni_dlams_dvertices(self, lams, n_vert):
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
        """The level indices making up a cluster of `mult` eigenvalues near `lam`: those of the
        members `lams` when given, else the adjacent levels whose multiplicities sum to `mult`."""
        if lams is not None:
            idx = sorted({self._level(l) for l in lams})
        else:
            lv = np.asarray(self.spec.lam)
            order = np.argsort(np.abs(lv - float(lam)))
            idx, got = [], 0
            for i in order:
                if got >= int(mult):
                    break
                idx.append(int(i))
                got += int(self.spec.mult[i])
            idx.sort()
        if sum(int(self.spec.mult[i]) for i in idx) != int(mult):
            raise ValueError(f"no set of levels near lam = {lam:.10g} holds exactly {mult} "
                             f"eigenvalues (levels {[float(self.spec.lam[i]) for i in idx]}, "
                             f"multiplicities {[int(self.spec.mult[i]) for i in idx]})")
        return idx

    def chladni_cluster_matrices(self, family, p, lam, mult, lams=None):
        """`(n_p, m, m)` for the cluster of `mult` eigenvalues near `lam`. One level of
        multiplicity `m`, or several distinct levels -- a nearly degenerate cluster -- taken as
        ONE block with their cross terms (`Spectrum.hadamard([i, j, ...])`)."""
        idx = self._cluster_levels(lam, mult, lams)
        key = tuple(idx)
        if key not in self._H:
            with _quiet():
                self._H[key] = self.spec.hadamard(idx[0] if len(idx) == 1 else idx,
                                                  self._velocity())
        H = self._H[key]                                        # (2N, m, m)
        Jv = family.d_vertices_dp(p)                            # (2N, n_p)
        return np.einsum("fij,fp->pij", H, Jv)


class ChladniEvaluator:
    """`SpectrumEvaluator`'s interface on the `chladni` backend. See the module docstring."""

    def __init__(self, family, n_eigs=1, prec=1e-10, plan_at=None, degenerate_tol=1e-6,
                 confirm=False):
        self.family = family
        self.n_eigs = int(n_eigs)
        self.prec = float(prec)
        self.degenerate_tol = float(degenerate_tol)
        self.confirm = bool(confirm)
        self._plan = None
        self._plan_at = None if plan_at is None else np.asarray(plan_at, dtype=float)
        self._plan_built_at = None
        self._last = None
        self.last_sigma = np.nan            # worst attained relative bound at the last shape
        self.n_replans = 0
        self.n_solver_refusals = 0
        self.refusals = []
        self.max_refusals_kept = 16

    def __repr__(self):
        return (f"ChladniEvaluator({self.family!r}, n_eigs={self.n_eigs}, "
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
        problem, as `SpectrumEvaluator.eigenvalues` does."""
        from chladni.eigenproblem import Eigenproblem
        dom, verts = self._domain(p)
        plan = self.plan(p)
        try:
            try:
                with _quiet():
                    spec = Eigenproblem(dom).solve(self.n_eigs, prec=self.prec, plan=plan)
            except ValueError as exc:
                if "corner" not in str(exc):
                    raise
                self.replan(p)                  # a different corner structure: re-plan here
                self.n_replans += 1
                with _quiet():
                    spec = Eigenproblem(dom).solve(self.n_eigs, prec=self.prec,
                                                   plan=self.plan(p))
        except (RuntimeError, ValueError) as exc:
            self.n_solver_refusals += 1
            self._record(p, f"{type(exc).__name__}: {exc}")
            raise SolverRefusal(f"chladni could not solve this shape: {exc}") from exc
        exc = refusal_for(spec)
        if exc is not None:
            self._record(p, str(exc))
            raise exc
        with np.errstate(invalid="ignore", divide="ignore"):
            self.last_sigma = float(np.nanmax(spec.bound / spec.lam)) if spec.lam.size else np.nan
        self._last = spec
        return spec.unfolded[:self.n_eigs], ChladniSolver(spec, verts, self.degenerate_tol)

    def _record(self, p, error):
        if len(self.refusals) < self.max_refusals_kept:
            self.refusals.append(dict(
                p=[float(t) for t in np.asarray(p, dtype=float)],
                plan_at=(None if self._plan_built_at is None
                         else [float(t) for t in self._plan_built_at]),
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
        """No warm start on this backend; kept for interface parity."""
        self._last = None
