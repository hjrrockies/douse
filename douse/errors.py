"""The refusals a forward solve can raise, shared by every backend.  (moved here 2026-10-14)

Kept in their own module so `evaluate` (which re-exports them) and `chladni_backend` can both
import them without an import cycle.
"""

__all__ = ['ShapeRefusal', 'SolverRefusal', 'UnresolvedSpectrum', 'DegenerateSpectrum']


class ShapeRefusal(RuntimeError):
    """The solver could not produce a usable spectrum at this shape.

    ONE class for every way a wild shape can defeat the forward solver, because the caller's
    response is the same for all of them: `maximize_ratio` treats the trial as UNREACHABLE --
    shrink the radius, count it, try a smaller step, stop on a streak -- exactly as it already
    treated `UnresolvedSpectrum`. Measured cost of not having it: five runs killed MID-RUN by
    `EigensolverFailure` and one by the `FundamentalBasis` interior-source guard, every one of
    them making progress at the time.

    The distinction the two subclasses keep is diagnostic, not procedural. `UnresolvedSpectrum`
    means a spectrum was computed and cannot be trusted to be `lam_1 .. lam_K`;
    `SolverRefusal` means no spectrum came back at all.
    """


class SolverRefusal(ShapeRefusal):
    """`lappy` declined to solve at this shape, and the shape is the reason.

    Wraps `EigensolverFailure` (`bracket_mins`: "f has too many local minima" -- a tension curve
    with no usable structure) and `SourcesInsideDomain` (the frozen plan's sources are no longer
    exterior, i.e. the shape has moved out of the plan's reach). The original is kept as
    `__cause__`, so a ledger row still records which one it was.

    Deliberately raised only from `eigenvalues`, where `p` names a specific trial shape. The same
    exception out of a fixed-shape call is a bug and should propagate.
    """


class UnresolvedSpectrum(ShapeRefusal):
    """The returned set is not `lam_1 .. lam_K`, and a full solve did not repair it.

    Raised rather than returned, because every consumer of `eigenvalues` reads the array as the
    consecutive spectrum and there is no value of it that is honest here. A caller that can
    treat a shape as unreachable -- `maximize_ratio` rejects the trial step and shrinks -- should
    catch this; a caller that cannot should let it propagate and record the traceback. Catch
    `ShapeRefusal` to cover this and `SolverRefusal` together, which is what both callers want.
    """


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
