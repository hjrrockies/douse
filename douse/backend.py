"""The seam between the optimizers and a forward solver.  (2026-10-14)

Two protocols and one record. `SpectrumBackend` is what `maximize_ratio`, `descend`,
`gauss_newton` and `ObjectiveFunction` consume: a family, a count, and five methods. It returns
the solver's `Spectrum` (levels with multiplicities, `chladni.eigensolver.Spectrum`) rather than
an array, because the optimizer's eps-active cluster must never split a level; and a
`SolvedShape`, which is what `douse.derivative` differentiates. `RunHealth` is the backend's
statistics as a VALUE returned per call, not state read off the evaluator by name: every
chladni ledger row once carried `nan` under names coined for lappy's tension, and fell out of
every summary, because the two backends kept different attributes.

Until today the seam was duck-typed: `derivative.py` dispatched on `hasattr`, `cluster_opt`
wrote `evaluator._last` to reseed warm tracking and read fourteen statistics through `getattr`
with defaults. `reseed` and `health` replace those.
"""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["SpectrumBackend", "SolvedShape", "RunHealth", "SolveTrace"]


@dataclass(frozen=True)
class SolveTrace:
    """One solve, as the backend saw it: `kind` is 'cold', 'warm' or 'replan'; `reason` why a
    requested warm solve went cold (chladni's `warm_note`) or why a replan was needed (the
    flags); `rung` the plan's ladder rung; `n_evals` the tension evaluations; `seconds` wall."""
    kind: str
    reason: object = None
    rung: object = None
    n_evals: int = 0
    seconds: float = float("nan")


@dataclass(frozen=True)
class RunHealth:
    """A backend's statistics, with chladni's meaning. Bounds, not tensions.

    `worst_bound`: running max of `last_bound`, the worst certified relative bound over the
    returned levels (near `prec` on a healthy run; the ledger gates at 1e-6). `worst_deficit`:
    running max of the count audit's mean Weyl deficit. `n_audit_flags`: audits that did not
    read "ok". Refusals by cause: `n_audit_refusals` (the count), `n_unmet_refusals` (a level
    unmet, uncertain or short), `n_solver_refusals` (no spectrum at all). `n_warm` /
    `n_warm_refused`: warm solves accepted / asked for and refused. `trace`: one `SolveTrace`
    per solve, capped."""
    worst_bound: float = float("-inf")
    last_bound: float = float("nan")
    worst_deficit: float = float("-inf")
    n_replans: int = 0
    n_solves: int = 0
    n_warm: int = 0
    n_warm_refused: int = 0
    n_tension_evals: int = 0
    n_audit_flags: int = 0
    n_audit_refusals: int = 0
    n_unmet_refusals: int = 0
    n_solver_refusals: int = 0
    min_gap: float = float("inf")
    refusals: tuple = ()
    plan_at: object = None
    plan_built_at: object = None
    trace: tuple = ()

    def as_row(self):
        """The ledger's keys (schema 2), JSON-ready."""
        def _f(x):
            x = float(x)
            return None if not np.isfinite(x) else x

        def _p(x):
            return None if x is None else [float(t) for t in np.asarray(x, dtype=float)]
        return dict(worst_bound=_f(self.worst_bound), last_bound=_f(self.last_bound),
                    worst_deficit=_f(self.worst_deficit), n_replans=int(self.n_replans),
                    n_solves=int(self.n_solves), n_warm=int(self.n_warm),
                    n_warm_refused=int(self.n_warm_refused),
                    n_tension_evals=int(self.n_tension_evals),
                    n_audit_flags=int(self.n_audit_flags),
                    n_audit_refusals=int(self.n_audit_refusals),
                    n_unmet_refusals=int(self.n_unmet_refusals),
                    n_solver_refusals=int(self.n_solver_refusals), min_gap=_f(self.min_gap),
                    refusals=list(self.refusals), plan_at=_p(self.plan_at),
                    plan_built_at=_p(self.plan_built_at),
                    trace=[dict(kind=t.kind, reason=t.reason, rung=t.rung, n_evals=t.n_evals,
                                seconds=_f(t.seconds)) for t in self.trace])


@runtime_checkable
class SolvedShape(Protocol):
    """One solved shape, as `douse.derivative` needs it."""

    def dlams_dvertices(self, lams, n_vert) -> np.ndarray:
        """`(K, 2N)`: the simple-eigenvalue gradients, `[x_0..x_{N-1}, y_0..y_{N-1}]`. Raises
        `DegenerateSpectrum` for a multiple level."""

    def cluster_matrices_dvertices(self, family, p, lam, mult, lams=None) -> np.ndarray:
        """`(n_p, m, m)`: the cluster block in the parameters. Raises `UnresolvedSpectrum` when
        the solver's levels do not hold exactly `mult` eigenvalues there."""


@runtime_checkable
class SpectrumBackend(Protocol):
    """A forward solver over a family of shapes, as the optimizers consume it."""

    family: object
    n_eigs: int
    degenerate_tol: float

    def eigenvalues(self, p):
        """`(spectrum, solved)`: a `Spectrum` (levels; `unfolded` is the array, with the level at
        `n_eigs` returned whole so `len >= n_eigs`) and a `SolvedShape`. Raises `ShapeRefusal`
        and nothing else when the shape is the problem."""

    def reseed(self, p, spectrum):
        """Point warm tracking at `(p, spectrum)`: the accepted iterate after a rejected
        trial, so the next solve continues from where the run is."""

    def reset_continuation(self):
        """Forget the previous shape; the next solve is cold."""

    def replan(self, p=None):
        """Throw the frozen plan away, to be rebuilt at `p` (or the next shape)."""

    def health(self) -> RunHealth:
        """The run's statistics so far, as a value."""
