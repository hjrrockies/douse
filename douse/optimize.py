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

__all__ = ['OptimizeResult', 'descend', 'gauss_newton']


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


def gauss_newton(evaluator, match, p0, max_iter=30, max_step=0.1, mu0=1e-3, grow=4.0,
                 shrink=0.25, rtol=1e-14, gtol=1e-12, max_backtrack=12, callback=None):
    """Levenberg-Marquardt on the inverse problem: `p -> ||match.residual(lams(p))||^2`.

    `descend` is a gradient method, and on a spectral match it is slow in the way gradient
    methods are on least squares: measured on a 1.2 x 0.8 rectangle recovered from a perturbed
    start, 150 iterations took the loss from 1.6e-04 to 6.7e-07 with the edges still 3% off.
    The residual is cheap to linearize -- `match.residual` gives `(r, J)` for the price of one
    spectrum and its Hadamard Jacobian -- and the Gauss-Newton step `(J'J + mu I) d = -J'r`
    converges quadratically on a consistent target.

    Every trial step is checked for admissibility BEFORE an eigensolve (`family.is_valid`), capped
    at `max_step` in `p` so the solver is never asked about a wild shape (one such trial cost
    an L-BFGS run nine minutes of refusal, 2026-10-14), and a `ShapeRefusal` at a trial point is
    treated as a rejected step: the damping grows and the step shortens.

    Returns an `OptimizeResult` with `x`, `fun` (the loss), `residual`, `nit`, `nfev`,
    `n_refused`, `success`, `message`. `success` is True on the residual test (`rtol`, on the
    loss) or the gradient test (`gtol`).
    """
    from .errors import ShapeRefusal

    is_valid = evaluator.family.is_valid
    K = match.n_eigs
    if K > evaluator.n_eigs:
        raise ValueError(f'{match!r} needs {K} eigenvalues but the evaluator supplies '
                         f'{evaluator.n_eigs}')

    def residual(q):
        lams, J = evaluator(q)
        return match.residual(np.asarray(lams)[:K], np.asarray(J)[:K])

    p = np.array(p0, dtype=float)
    if not is_valid(p):
        raise ValueError('p0 is not an admissible parameter vector')
    r, J = residual(p)
    loss = float(r @ r)
    nfev, n_refused, mu = 1, 0, float(mu0)
    success, message, it = False, f'reached max_iter={max_iter}', 0
    for it in range(1, max_iter + 1):
        g = J.T @ r
        if loss < rtol:
            success, message = True, f'loss {loss:.3g} below rtol={rtol:g}'
            break
        if float(np.linalg.norm(g)) < gtol:
            success, message = True, f'gradient norm {np.linalg.norm(g):.3g} below gtol={gtol:g}'
            break
        stepped = False
        for _ in range(max_backtrack):
            A = J.T @ J + mu * np.eye(len(p))
            d = -np.linalg.solve(A, g)
            n = float(np.linalg.norm(d))
            if n > max_step:
                d *= max_step / n
            q = p + d
            if is_valid(q):
                try:
                    r2, J2 = residual(q)
                    nfev += 1
                except ShapeRefusal:
                    n_refused += 1
                    r2 = None
                if r2 is not None:
                    loss2 = float(r2 @ r2)
                    if np.isfinite(loss2) and loss2 < loss:
                        p, r, J, loss, stepped = q, r2, J2, loss2, True
                        mu = max(mu * shrink, 1e-15)
                        break
            mu *= grow
        if not stepped:
            message = (f'no decrease after {max_backtrack} damped steps (mu {mu:.3g}); the '
                       f'residual is probably at the spectrum\'s own accuracy')
            break
        if callback is not None:
            callback(p, loss, r)
    return OptimizeResult(x=p, fun=float(loss), residual=r, jac=J.T @ r, nit=it, nfev=nfev,
                          n_refused=n_refused, grad_norm=float(np.linalg.norm(J.T @ r)),
                          success=success, message=message)
