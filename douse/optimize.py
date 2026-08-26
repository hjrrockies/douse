"""Driving an objective downhill, over a parametrization whose domain has a boundary.

WHY NOT JUST CALL SCIPY. The parametrization's admissible set is not all of `R^(2N-4)`: a `p`
whose free vertices spend the entire perimeter budget describes no polygon. `ObjectiveFunction`
returns `inf` there, which is the conventional signal, and `scipy.optimize.minimize`'s L-BFGS-B
handles it badly -- measured, it proposed an inadmissible point on its FIRST line search, took
the `inf`, and declared convergence at iteration 1 having moved nowhere. An `inf` tells a line
search that a step failed but gives it no direction, so a quasi-Newton method that lands outside
tends to stop rather than recover.

So the search here backtracks explicitly, testing admissibility BEFORE spending an eigensolve on
a point. That is cheap (`is_valid` is pure geometry) and it means the expensive path is only ever
walked on shapes that exist.

This is a deliberately plain method -- backtracking descent with an adaptive step -- and it is
enough to recover the known optima to 1e-10 in `lam`. It is not enough for the problems this
package is ultimately for: at a degenerate optimum `lam` is only directionally differentiable and
no gradient method converges, which is the open question recorded in `EigenvalueObjective`.
"""
import numpy as np

__all__ = ['OptimizeResult', 'descend']


class OptimizeResult(dict):
    """Attribute-accessible result, in the shape scipy uses."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__

    def __repr__(self):
        # BUILT FROM WHATEVER IS PRESENT, because two producers fill this in differently:
        # `descend` sets `nfev` and `grad_norm`, `cluster_opt.maximize_ratio` sets neither (it has
        # no line search and no gradient at a cluster) and sets `delta`, `gap` and `worst_sigma`
        # instead. The fixed format string raised `AttributeError: nfev` on every
        # `maximize_ratio` result, so simply printing one -- in a notebook, in a log line, in a
        # debugger -- failed.
        parts = []
        for key, fmt in (('fun', '.12g'), ('nit', 'd'), ('nfev', 'd'), ('grad_norm', '.3g'),
                         ('gap', '.3g'), ('delta', '.3g'), ('worst_sigma', '.1e')):
            if key in self and self[key] is not None:
                try:
                    parts.append(f'{key}={self[key]:{fmt}}')
                except (TypeError, ValueError):
                    parts.append(f'{key}={self[key]!r}')
        for key in ('success', 'message'):
            if key in self:
                parts.append(f'{key}={self[key]!r}')
        return f"OptimizeResult({', '.join(parts)})"


def descend(obj, p0, is_valid=None, max_iter=200, alpha0=0.02, grow=1.6, shrink=0.5,
            max_backtrack=40, gtol=1e-7, ftol=0.0, callback=None):
    """Backtracking gradient descent on `obj(p) -> (value, gradient)`.

    Parameters
    ----------
    obj : callable
        Returns `(value, gradient)`. A `douse.evaluate.ObjectiveFunction` fits directly.
    p0 : array
        Starting parameters. Must be admissible.
    is_valid : callable, optional
        `p -> bool`, checked before each trial step so that an inadmissible point costs no
        eigensolve. Defaults to `obj.evaluator.family.is_valid` when `obj` provides it.
    alpha0, grow, shrink : float
        Initial step, and the factors applied after a success and after a rejection. Growing on
        success is what keeps the iteration count reasonable: the step needed near the start and
        the step needed near the optimum differ by orders.
    gtol : float
        Stop when `|grad| < gtol`.
    ftol : float
        Stop when a successful step improves the objective by less than this (relative). Zero
        disables it, which is usually right here -- the objective is an eigenvalue and its own
        accuracy is the real floor.

    Returns
    -------
    OptimizeResult with `x`, `fun`, `jac`, `nit`, `nfev`, `grad_norm`, `success`, `message`.

    Stopping is reported honestly: `success` is True only for the gradient test. Exhausting
    `max_iter`, or backtracking `max_backtrack` times without finding a decrease, both return
    `success=False` with a message saying which -- the second usually means the step has shrunk
    below what the objective's own accuracy can resolve, not that a minimum was reached.
    """
    if is_valid is None:
        is_valid = getattr(getattr(obj, 'evaluator', None), 'family', None)
        is_valid = is_valid.is_valid if is_valid is not None else (lambda p: True)

    p = np.array(p0, dtype=float)
    if not is_valid(p):
        raise ValueError('p0 is not an admissible parameter vector')

    val, grad = obj(p)
    if not np.isfinite(val):
        raise ValueError(f'the objective is not finite at p0 (got {val})')

    nfev, alpha, message = 1, float(alpha0), f'reached max_iter={max_iter}'
    success, it = False, 0

    for it in range(1, max_iter + 1):
        gnorm = float(np.linalg.norm(grad))
        if gnorm < gtol:
            success, message = True, f'gradient norm {gnorm:.3g} below gtol={gtol:g}'
            break

        stepped = False
        for _ in range(max_backtrack):
            q = p - alpha*grad
            if is_valid(q):
                v2, g2 = obj(q)
                nfev += 1
                if np.isfinite(v2) and v2 < val:
                    improvement = (val - v2)/max(abs(val), 1.0)
                    p, val, grad, stepped = q, v2, g2, True
                    alpha *= grow
                    if ftol > 0 and improvement < ftol:
                        success = True
                        message = f'improvement {improvement:.3g} below ftol={ftol:g}'
                    break
            alpha *= shrink
        if not stepped:
            message = (f'no decrease after {max_backtrack} backtracks (step {alpha:.3g}); the '
                       f'step is probably below what the objective can resolve')
            break
        if callback is not None:
            callback(p, val, grad)
        if success:
            break

    return OptimizeResult(x=p, fun=float(val), jac=grad, nit=it, nfev=nfev,
                          grad_norm=float(np.linalg.norm(grad)), success=success,
                          message=message)
