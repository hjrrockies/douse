"""Turning a parameter vector into a spectrum and its Jacobian.

This is the inner loop. It owns the two decisions that make a shape-optimization run cheap or
expensive, and correct or subtly wrong:

**Plan once, realize per iterate.** `lappy.basis_plan` splits deciding a basis (term counts, arc
endpoints, as fractions) from building it (positions). The plan is frozen at the starting shape
and only realized on each new one, so `n_basis` is identical at every iterate and a change in
`lam` came from the SHAPE rather than from the discretization. Re-planning every iterate would
change the basis size underneath the optimizer -- which is not merely wasteful, it makes the
objective discontinuous.

**Track by value, then RESTORE THE ORDER.** Each iterate is seeded from the previous one, which
is ~2.7x faster than a fresh global scan. But tracking follows a mode THROUGH a crossing, while
an objective written in terms of `lam_k` means the k-th SMALLEST -- so the tracked array must be
sorted before anything indexes it.

Omitting that sort is not a cosmetic bug. Maximizing `lam_2/lam_1` over quadrilaterals reported
2.6936 at its third iterate, above the Payne--Polya--Weinberger bound of 2.5387 that every
bounded planar domain obeys. Tracking had returned `[19.855, 50.807, 48.444]`: `lam_2` and
`lam_3` had crossed and swapped, so the objective read the third mode as the second. The true
ratio there was 2.2877. Nothing about the number looked wrong except that it violated a theorem.

Sorting alone is not enough either, because two seeds can converge onto the SAME mode, which
leaves a duplicate and a missing eigenvalue that sorting cannot detect. `_verify_multiplicity`
checks any coincident group against the tension spectrum -- a genuine m-fold eigenvalue has m
small tensions, a collapse has one -- and falls back to a full solve when the check fails.

**...and check that what tracking returned is actually an eigenvalue.** `lappy`'s `track` refuses
a minimum whose tension exceeds `ttol`, but the default `ttol = 1e-3` is calibrated for FINDING
eigenvalues, not for the precision an optimizer needs. Measured, on the first descent run through
this class: `track` returned a point with `sigma = 2.0e-04` -- inside `ttol`, so no refusal --
where the true eigenvalue was 2.9e-03 away, against a basis whose tension floor at that very
shape is 5e-12. The optimizer then descended 2.8e-03 BELOW the true minimum, which for `min lam_1`
over quadrilaterals is a provably impossible answer, and would have been an invisible one for any
objective without a known optimum.

So a tracked eigenvalue is VERIFIED here, against `track_ttol` (1e-8 by default), and a failure
falls back to a full solve.

The verification is separate rather than a tighter `ttol` on the solver, which was tried first
and is wrong: `ttol` gates two different questions. `solve_interval` uses it to decide whether a
tension minimum is an eigenvalue AT ALL, and `track` uses it to decide whether a located value is
converged. Tightening it to 1e-8 made the second work and broke the first -- `solve(1)` returned
ZERO eigenvalues on a perfectly ordinary triangle, because genuine modes whose tension sits
above 1e-8 were being filtered out. 1e-3 is a reasonable threshold for detection and a wholly
unreasonable one for convergence, so the two get their own numbers.
"""
import warnings

import numpy as np

from lappy import Eigenproblem, basis_plan as BP
from lappy.asymp import weyl_est
from lappy.core import EigensolverFailure
from lappy.mps import MPSEigensolver

from .derivative import dlams_dp

__all__ = ['SpectrumEvaluator', 'ObjectiveFunction', 'DegenerateSpectrum']


class DegenerateSpectrum(RuntimeError):
    """Two tracked eigenvalues have merged, so the simple-eigenvalue gradient is meaningless.

    At a multiple eigenvalue `lam` is only DIRECTIONALLY differentiable: the modes that emerge
    under a perturbation `V` are the eigenvalues of `M(V)_ij = -int (du_i/dn)(du_j/dn)(V.n) ds`,
    and `lappy.weighted_integral(..., mult=m)` returns exactly that matrix. What the
    simple-eigenvalue path returns instead is built from `eigenfunction_coef(lam, mult=1)`, which
    hands back an ARBITRARY vector out of the m-dimensional eigenspace -- so it is not a bad
    approximation to the right answer, it is a different quantity.

    Measured at the unit square, where lam_2 = lam_3 = 5 pi^2 exactly:

        parameter      true directional derivatives      simple-eigenvalue path
        p[0]              -14.222,  +14.222                  +14.032,  +14.032
        p[2]              -32.847,  +32.847                   +9.212,   +9.212
        p[3]              -29.609,  +29.609                   -4.820,   -4.820

    Identical for both modes, wrong by factors of three to six, and carrying no sign that two
    branches exist at all. Hence a raise rather than a warning: an optimizer that continued here
    would follow a confident wrong gradient, and in a long run a warning is just noise.

    Note the true splits are symmetric about zero -- `tr M = 0` in these directions -- which is
    the general fact that SYMMETRIC functions of a cluster stay smooth even where the individual
    eigenvalues do not. `lam_2 + lam_3` has an honest gradient at the square. Rephrasing an
    objective in terms of the whole cluster is often cheaper than handling the nonsmoothness.
    """


class SpectrumEvaluator:
    """`p -> (lam_1..lam_K, d lam/dp)` for a fixed family, on a frozen basis plan.

        ev = SpectrumEvaluator(VertexPolygon(4, perim=4.0), n_eigs=3)
        lams, J = ev(p)

    `target` sizes the basis and `prec` the eigenvalue search. The defaults are tighter than
    `lappy`'s, because a finite-difference check of anything built on this is limited by the
    tension floor rather than by the search tolerance -- see `tests/test_derivative.py`.
    """

    def __init__(self, family, n_eigs=1, target=1e-14, prec=1e-13, plan_at=None,
                 lam_max_window=6, warn_gap=1e-3, degenerate_tol=1e-6,
                 track_ttol=1e-8):
        self.family = family
        self.n_eigs = int(n_eigs)
        self.target = float(target)
        self.prec = float(prec)
        self.lam_max_window = int(lam_max_window)
        self.warn_gap = float(warn_gap)
        self.degenerate_tol = float(degenerate_tol)
        self.track_ttol = float(track_ttol)
        self._plan = None
        self._plan_at = None if plan_at is None else np.array(plan_at, dtype=float)
        self._last = None            # (p, lams) for continuation
        self.n_solves = 0
        self.n_tracks = 0
        self.n_track_refusals = 0
        self.n_collapses = 0
        self.min_gap = np.inf

    def __repr__(self):
        return (f'SpectrumEvaluator({self.family!r}, n_eigs={self.n_eigs}, '
                f'target={self.target:.0e})')

    def plan(self, p):
        """The frozen `BasisPlan`, built on first use at `plan_at` (default: the first `p` seen).

        Deliberately never rebuilt. If the shape wanders far enough that the plan no longer
        resolves it, that shows up as a rising tension rather than as a silent change of basis,
        and `sigma_at` is how a caller checks.
        """
        if self._plan is None:
            at = self._plan_at if self._plan_at is not None else np.asarray(p, dtype=float)
            dom = self.family(at)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._plan = BP.plan_basis(dom, weyl_est(self.lam_max_window, dom),
                                           target=self.target)
        return self._plan

    def solver(self, p):
        """A solver for `family(p)`, on the frozen plan."""
        dom = self.family(p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return dom, MPSEigensolver.from_domain(
                dom, lam_max=weyl_est(self.lam_max_window, dom),
                basis=BP.realize(self.plan(p), dom), prec=self.prec)

    def eigenvalues(self, p, seed=None):
        """`(lams, solver)`. Tracks from the previous iterate when there is one.

        Falls back to a full `solve` if tracking refuses -- which it does rather than returning a
        window edge or a point where the tension is too large. A refusal means the shape moved
        further than the scan window, so the honest response is to pay for a global scan, not to
        widen the window until something is found.
        """
        p = np.asarray(p, dtype=float)
        dom, solver = self.solver(p)
        evp = Eigenproblem(dom, eval_solver=solver, precision=self.prec)

        if seed is None and self._last is not None:
            seed = self._last[1]

        lams = None
        if seed is not None and len(seed) >= self.n_eigs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cand = np.sort(np.array([evp.track(float(s))
                                             for s in seed[:self.n_eigs]]))
                    worst = max(float(np.atleast_1d(solver.sigma(float(l)))[0]) for l in cand)
                    collapsed = not self._verify_multiplicity(solver, cand)
                if worst > self.track_ttol:
                    # `track`'s own guard uses lappy's ttol=1e-3, which is a detection threshold,
                    # not a convergence one. Verify against something an optimizer can use.
                    self.n_track_refusals += 1
                elif collapsed:
                    self.n_collapses += 1        # two seeds landed on one mode
                else:
                    lams = cand
                    self.n_tracks += 1
            except EigensolverFailure:
                self.n_track_refusals += 1       # fall through to a full solve
        if lams is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lams = np.asarray(evp.solve(self.n_eigs), dtype=float)[:self.n_eigs]
            self.n_solves += 1

        if len(lams) > 1:
            gap = float(np.min(np.diff(np.sort(lams))/np.sort(lams)[:-1]))
            self.min_gap = min(self.min_gap, gap)
            if gap < self.degenerate_tol:
                raise DegenerateSpectrum(
                    f'two of the tracked eigenvalues are within {gap:.2e} (relative) at this '
                    f'shape, below degenerate_tol={self.degenerate_tol:.1e}. The '
                    f'simple-eigenvalue gradient is not an approximation here, it is a different '
                    f'quantity -- see DegenerateSpectrum. Use the m x m cluster matrix, or an '
                    f'objective that is a symmetric function of the cluster, or raise '
                    f'degenerate_tol if you know this crossing is benign.')
            if gap < self.warn_gap:
                warnings.warn(
                    f'the tracked eigenvalues are within {gap:.2e} (relative) of each other at '
                    f'this shape. An index-based objective is not differentiable through a '
                    f'crossing, and these problems tend to optimize TOWARDS one, so a gradient '
                    f'method may stall or chatter here rather than converge.')

        self._last = (p.copy(), lams.copy())
        return lams, solver

    def _verify_multiplicity(self, solver, lams, rtol=1e-9):
        """Does each coincident group of tracked values really have that multiplicity?

        Two seeds converging onto one mode leaves a duplicate and drops an eigenvalue, and no
        amount of sorting reveals it -- the array looks like a legitimate degenerate spectrum.
        The tension spectrum tells them apart: an m-fold eigenvalue makes the first m generalized
        singular values small, a single mode makes only the first one small.
        """
        i = 0
        while i < len(lams):
            j = i + 1
            while j < len(lams) and (lams[j] - lams[j - 1]) <= rtol*abs(lams[j]):
                j += 1
            m = j - i
            if m > 1:
                t = np.atleast_1d(solver.tensions(float(np.mean(lams[i:j]))))
                if len(t) < m or t[m - 1] > self.track_ttol:
                    return False
            i = j
        return True

    def __call__(self, p, seed=None):
        """`(lams, dlams_dp)` -- the spectrum and its `(K, n_p)` Jacobian."""
        lams, solver = self.eigenvalues(p, seed=seed)
        return lams, dlams_dp(self.family, p, solver, lams)

    def sigma_at(self, p, lam):
        """The MPS tension at `(p, lam)`: how well the frozen plan still resolves this shape."""
        _, solver = self.solver(p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(np.atleast_1d(solver.sigma(float(lam)))[0])

    def reset_continuation(self):
        """Forget the previous iterate, so the next call does a full solve. Use after a jump."""
        self._last = None


class ObjectiveFunction:
    """`p -> (value, gradient)`, ready for `scipy.optimize.minimize(..., jac=True)`.

        obj = ObjectiveFunction(SpectrumEvaluator(fam, n_eigs=1), EigenvalueObjective(0))
        val, grad = obj(p)

    Counts its own evaluations and records the history, because the first thing anyone asks of a
    shape optimization is what it actually did.

    OUT-OF-RANGE PARAMETERS ARE A FACT OF LIFE HERE, not an error. The parametrization's domain
    is not all of R^(2N-4): a `p` whose free vertices spend the whole perimeter budget describes
    no polygon at all, and an unconstrained line search WILL propose one -- measured, on the
    first optimization ever run through this class, at iteration one. So an inadmissible `p`
    returns `inf` with a zero gradient rather than raising, which is the convention a line search
    understands: it backs off and retries shorter.

    That is a stopgap and not a constraint. `inf` gives a line search no direction, so an
    optimizer that wanders far outside will stall rather than be guided back, and nothing here
    prevents a self-intersecting polygon that IS admissible to the perimeter arithmetic. A real
    barrier -- the smooth self-intersection constraint sketched in the pre-Milestone-1 `opt.py`,
    plus one for the closing budget -- belongs with the optimizer, in Milestone 3.
    """

    def __init__(self, evaluator, objective, invalid_value=np.inf):
        if objective.n_eigs > evaluator.n_eigs:
            raise ValueError(f'{objective!r} needs {objective.n_eigs} eigenvalues but the '
                             f'evaluator supplies {evaluator.n_eigs}')
        self.evaluator = evaluator
        self.objective = objective
        self.invalid_value = float(invalid_value)
        self.history = []
        self.n_invalid = 0

    def __repr__(self):
        return f'ObjectiveFunction({self.evaluator!r}, {self.objective!r})'

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if not self.evaluator.family.is_valid(p):
            self.n_invalid += 1
            self.history.append(dict(p=p.copy(), lams=None, value=self.invalid_value,
                                     grad_norm=0.0, valid=False))
            return self.invalid_value, np.zeros_like(p)

        lams, dlams = self.evaluator(p)
        val, grad = self.objective(lams, dlams)
        self.history.append(dict(p=p.copy(), lams=lams.copy(), value=float(val),
                                 grad_norm=float(np.linalg.norm(grad)), valid=True))
        return float(val), np.asarray(grad, dtype=float)
