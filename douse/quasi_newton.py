"""A second-order model for the SMOOTH stretches of a ratio maximization.

WHY. `cluster_opt`'s smooth branch was steepest ascent inside a trust region --
`d = Delta * g/||g||` -- which ignores curvature entirely and always sits on the trust boundary.
That is the right thing to do when no curvature is known and the wrong thing to do for the
several hundred iterations a run spends away from any degeneracy. Steepest ascent on an
ill-conditioned quadratic converges at the rate `((k-1)/(k+1))^2` per step in the condition
number `k`, which is why `max lam_3/lam_1` at N=5 needed more than 250 iterations at `n_p = 6`
and did not converge. N=16 is `n_p = 28`.

WHAT IS HERE. A BFGS approximation `B` to the Hessian of `-F` accumulated from the gradients
the optimizer already computes, and an EXACT solve of the resulting trust-region subproblem

    minimize   g.d + (1/2) d^T B d      subject to   ||d|| <= Delta

by the More--Sorensen characterization: `d(sigma) = -(B + sigma I)^{-1} g` for the unique
`sigma >= max(0, -lam_min(B))` making `||d|| = Delta`, or the interior Newton step when that is
already short enough. `n_p` is at most 28 here, so the eigendecomposition this is built on costs
nothing next to a ~300 ms eigensolve, and there is no reason to settle for the dogleg or for
Steihaug-CG's truncation.

THE PAIRS COME ONLY FROM ACCEPTED SMOOTH STEPS. Two rules, and both matter:

  * `y` is a difference of gradients of `F`, and at a cluster `F` has no gradient. A pair
    straddling a cluster excursion is a difference of two things that are not comparable, so the
    chain is BROKEN whenever the model switches to cluster mode -- see `chain_broken`. The matrix
    itself is KEPT as a warm start, because the curvature of the smooth part does not stop
    existing while the optimizer passes a crossing.
  * A REJECTED step yields no gradient at the trial point (the run evaluates `F` there but not
    `dF`), so only accepted steps produce pairs.

CURVATURE IS SKIPPED, NOT DAMPED, WHEN `s.y <= 0`. `F` is not concave -- these runs climb toward
saddle-shaped ridges and through them -- so `s.y <= 0` is a fact about the landscape rather than
a numerical accident, and Powell damping would manufacture curvature that is not there. Skipping
leaves `B` stale for one step; the trust region is what keeps the step honest either way, and the
exact subproblem solve below handles an indefinite `B` correctly if one ever arises.

THE STOPPING CERTIFICATE IS UNCHANGED. `maximize_ratio` measures criticality against the
FIRST-ORDER model, which in smooth mode is exactly `||grad F||` whatever the step was. A
quadratic model's gain per unit radius is not a criticality measure: near a maximum the Newton
step goes interior and its gain falls quadratically in `||grad||` while `Delta` stays at
`delta_max`, so `gain/Delta` would declare optimality at `||grad|| ~ 3e-04`. The quadratic gain
is used for `rho` -- where it belongs, and where it is what makes `rho -> 1` hold to second
order -- and never for the stop.

SR1 BLOWS UP ON THESE RUNS, AND THE TRUST REGION CANNOT SAVE IT (measured 2026-10-15, N=6
`lam_3/lam_1` seed 0, both warm paths). The SR1 matrix for `lam_1` grew from a norm of 1.5e2 at
iterate 18 to 1e7 by iterate 60 against a true Hessian of norm 2.2e2 (central differences of
the Hadamard gradient, symmetric to 3e-7, so the gradients are not the problem); the cluster
mean's followed, to 6e11. Mechanism: once the steps zigzag inside a low-dimensional subspace,
`s.(y - Bs)` cancels while `|y - Bs|` stays at the size of the third-order terms, so each
rank-one update is enormous and the 1e-8 skip rule never fires. The second-order gain then
over- or under-shoots the first-order one by up to five orders, `rho` decays to the acceptance
threshold, rejections halve the radius, and nothing repairs the matrix because rejected trials
produce no pairs: the "trust radius collapsed to 5.68e-15" ending. A cap on each update
relative to the current norm compounds and does not help. `ScalarSecant` is the default now:
the curvature along the last step, a Rayleigh quotient that cannot leave the true Hessian's
range. Over 100 iterates from a healthy state it matched SR1 to 5e-6 in the objective, so the
directional information SR1 carried was worth nothing here.
"""
import numpy as np

__all__ = ['BFGS', 'SR1', 'ScalarSecant', 'tr_subproblem']


def tr_subproblem(g, B, delta, tol=1e-12, maxiter=200):
    """`argmin_{||d|| <= delta} g.d + (1/2) d^T B d`, exactly.

    Handles an indefinite `B` and the hard case (`g` orthogonal to the lowest eigenspace, with
    the boundary not reached from `sigma = -lam_min`), where the solution picks up a component
    along that eigenspace. Returns `(d, predicted_decrease)` with the decrease non-negative.
    """
    g = np.asarray(g, dtype=float)
    n = g.size
    delta = float(delta)
    w, V = np.linalg.eigh(np.asarray(B, dtype=float))
    gt = V.T @ g

    def d_of(sigma):
        return -(V @ (gt/(w + sigma)))

    # Interior solution: available only when B is positive definite and the Newton step fits.
    if w[0] > 0:
        d = d_of(0.0)
        if np.linalg.norm(d) <= delta:
            return d, _decrease(g, B, d)

    lo = max(0.0, -w[0])
    # Is the lowest eigenspace excited? If not, and the boundary is not reached at sigma = lo,
    # this is the hard case.
    near_lo = np.abs(w - w[0]) <= 1e-12*max(1.0, abs(w[0]))
    excited = float(np.linalg.norm(gt[near_lo])) > tol*max(1.0, float(np.linalg.norm(gt)))

    if not excited:
        safe = ~near_lo
        d = np.zeros(n)
        if np.any(safe):
            coef = np.zeros(n)
            coef[safe] = -gt[safe]/(w[safe] + lo)
            d = V @ coef
        nd = float(np.linalg.norm(d))
        if nd <= delta:
            tau = np.sqrt(max(delta**2 - nd**2, 0.0))
            d = d + tau*V[:, 0]
            return d, _decrease(g, B, d)

    # Boundary solution: ||d(sigma)|| decreases monotonically from +inf (or from its value at
    # `lo`) to 0, so bracket by doubling and bisect. A few dozen halvings is exact to 1e-15
    # relative and costs nothing at these sizes.
    a = lo + 1e-14*max(1.0, abs(w[0]))
    while np.linalg.norm(d_of(a)) < delta and a > lo:
        a = lo + (a - lo)/10
        if a - lo < 1e-300:
            break
    b = max(a, 1.0)
    while np.linalg.norm(d_of(b)) > delta and b < 1e300:
        b *= 2.0
    for _ in range(maxiter):
        mid = 0.5*(a + b)
        if np.linalg.norm(d_of(mid)) > delta:
            a = mid
        else:
            b = mid
        if b - a <= tol*max(1.0, b):
            break
    d = d_of(0.5*(a + b))
    nd = float(np.linalg.norm(d))
    if nd > delta and nd > 0:
        d = d*(delta/nd)
    return d, _decrease(g, B, d)


def _decrease(g, B, d):
    """The model's predicted DECREASE, `-(g.d + 0.5 d^T B d)`, floored at zero."""
    return float(max(0.0, -(g @ d + 0.5*(d @ (B @ d)))))


class BFGS:
    """A BFGS approximation to the Hessian of the MINIMIZED objective, `-F`.

    `update(s, y)` takes `s = p_new - p_old` and `y = g_new - g_old` with `g` the gradient of
    `-F`. The first accepted pair also sets the scale, `B = (y.y)/(s.y) I` -- without it the
    identity start is off by whatever `lam` happens to be worth in these units (`|F| ~ 3`,
    curvature `~ 10^1-10^2`), and the first several steps are spent discovering that.
    """

    def __init__(self, n):
        self.n = int(n)
        self.B = np.eye(self.n)
        self.n_updates = 0
        self.n_skipped = 0
        self.chain = None            # (p, g) from the previous accepted smooth iterate

    def __repr__(self):
        return f'BFGS(n={self.n}, updates={self.n_updates}, skipped={self.n_skipped})'

    @property
    def ready(self):
        """Has any curvature been absorbed? Until then the step is steepest descent anyway."""
        return self.n_updates > 0

    def reset(self):
        self.B = np.eye(self.n)
        self.n_updates = 0
        self.chain = None

    def chain_broken(self):
        """Forget the previous iterate without forgetting the curvature. See the module docstring."""
        self.chain = None

    def update(self, s, y):
        """Absorb one `(s, y)` pair. Returns True if the curvature condition let it through."""
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        sy = float(s @ y)
        ns, ny = float(np.linalg.norm(s)), float(np.linalg.norm(y))
        if not np.isfinite(sy) or ns == 0 or ny == 0 or sy <= 1e-8*ns*ny:
            self.n_skipped += 1
            return False
        if self.n_updates == 0:
            self.B = ((y @ y)/sy)*np.eye(self.n)
        Bs = self.B @ s
        sBs = float(s @ Bs)
        if sBs <= 0:
            self.n_skipped += 1
            return False
        self.B = self.B + np.outer(y, y)/sy - np.outer(Bs, Bs)/sBs
        self.B = 0.5*(self.B + self.B.T)
        self.n_updates += 1
        return True

    def observe(self, p, g):
        """Record `(p, g)` at an accepted smooth iterate, updating from the previous one.

        Returns True if a pair was formed and absorbed.
        """
        p = np.asarray(p, dtype=float).copy()
        g = np.asarray(g, dtype=float).copy()
        took = False
        if self.chain is not None:
            took = self.update(p - self.chain[0], g - self.chain[1])
        self.chain = (p, g)
        return took


class ScalarSecant:
    """`B = gamma I` with `gamma = s.y / s.s`, the curvature along the last accepted step.

    The Rayleigh quotient of the average Hessian over the step, so `gamma` lies in the range of
    the true Hessian's eigenvalues whatever the pairs do -- the property SR1 lacks (module
    docstring). It carries one number: the curvature in the direction the run is walking, which
    is the only direction the trust-region step of the next iterate differs from the first-order
    one by. Same interface as `SR1` so `cluster_opt` and `model_gain2` need not know which."""

    def __init__(self, n):
        self.n = int(n)
        self.B = np.zeros((self.n, self.n))
        self.gamma = 0.0
        self.n_updates = 0
        self.n_skipped = 0
        self.chain = None

    def __repr__(self):
        return f'ScalarSecant(n={self.n}, gamma={self.gamma:.3g}, updates={self.n_updates})'

    @property
    def ready(self):
        return self.n_updates > 0

    def reset(self):
        self.B = np.zeros((self.n, self.n))
        self.gamma = 0.0
        self.n_updates = 0
        self.chain = None

    def chain_broken(self):
        """Forget the previous iterate, keep the curvature."""
        self.chain = None

    def update(self, s, y):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        ss, sy = float(s @ s), float(s @ y)
        if ss == 0 or not np.isfinite(sy):
            self.n_skipped += 1
            return False
        self.gamma = sy/ss
        self.B = self.gamma*np.eye(self.n)
        self.n_updates += 1
        return True

    def observe(self, p, g):
        """Record `(p, g)`; update from the previous recorded pair if the chain is unbroken."""
        p = np.asarray(p, dtype=float).copy()
        g = np.asarray(g, dtype=float).copy()
        took = False
        if self.chain is not None:
            took = self.update(p - self.chain[0], g - self.chain[1])
        self.chain = (p, g)
        return took


class SR1:
    """A symmetric-rank-one approximation to the Hessian of a scalar that need not be convex.

    WHY NOT BFGS HERE. `BFGS` above keeps its matrix positive definite, which is the right thing
    for a function being minimized to a minimum and the WRONG thing for `lam_1(p)` and for the
    cluster mean `mu(p)`. Neither is convex along the path these runs take -- they climb toward
    saddle-shaped ridges -- so BFGS would spend most of its pairs in the skip branch and report
    curvature it had refused to learn. SR1 has no definiteness requirement: it converges to the
    true Hessian, sign and all, and the trust-region subproblem in `tr_subproblem` solves the
    indefinite case exactly, so there is nothing downstream that needs it tamed.

    The one thing SR1 does need is the standard safeguard against a vanishing denominator, which
    is `|s.(y - Bs)| >= r ||s|| ||y - Bs||`. A skipped update costs one step of staleness; an
    unskipped one with a denominator near zero destroys the matrix.
    """

    def __init__(self, n, r=1e-8):
        self.n = int(n)
        self.r = float(r)
        self.B = np.zeros((self.n, self.n))
        self.n_updates = 0
        self.n_skipped = 0
        self.chain = None            # (p, g) at the previous iterate this scalar was smooth on

    def __repr__(self):
        return f'SR1(n={self.n}, updates={self.n_updates}, skipped={self.n_skipped})'

    @property
    def ready(self):
        """Zero until a pair is absorbed, which makes the model exactly the first-order one."""
        return self.n_updates > 0

    def reset(self):
        self.B = np.zeros((self.n, self.n))
        self.n_updates = 0
        self.chain = None

    def chain_broken(self):
        """Forget the previous iterate, keep the curvature. See `cluster_opt` for when."""
        self.chain = None

    def update(self, s, y):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        w = y - self.B @ s
        den = float(s @ w)
        nw, ns = float(np.linalg.norm(w)), float(np.linalg.norm(s))
        if not np.isfinite(den) or ns == 0 or nw == 0 or abs(den) < self.r*ns*nw:
            self.n_skipped += 1
            return False
        self.B = self.B + np.outer(w, w)/den
        self.B = 0.5*(self.B + self.B.T)
        self.n_updates += 1
        return True

    def observe(self, p, g):
        """Record `(p, g)`; update from the previous recorded pair if the chain is unbroken."""
        p = np.asarray(p, dtype=float).copy()
        g = np.asarray(g, dtype=float).copy()
        took = False
        if self.chain is not None:
            took = self.update(p - self.chain[0], g - self.chain[1])
        self.chain = (p, g)
        return took
