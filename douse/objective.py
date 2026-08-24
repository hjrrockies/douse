"""Objectives on a spectrum, and their gradients.

Everything here is a function of the eigenvalues alone -- pure arithmetic on a `(K,)` vector and
its `(K, n_p)` Jacobian, with no geometry and no solver in sight. That separation is deliberate:
these are the pieces most likely to be swapped out as the science changes, and they are the
cheapest to test, so they should not need an eigensolve to exercise.

TWO OBJECTIVES, for the two problems.

`EigenvalueObjective` minimizes a single eigenvalue. The perimeter is already pinned by the
parametrization, so `min lam_1` over perimeter-1 N-gons is well posed as it stands. Its answers
are known for `n = 3, 4` -- the equilateral triangle and the square -- which is what makes it the
right thing to validate the machinery on before pointing it at anything open.

`SpectralMatch` is the inverse problem: given a target spectrum from a domain of unknown SIZE,
find the shape. Absolute eigenvalues cannot be matched without knowing the scale, so it compares
scale-invariant coordinates instead -- see `normalized_reciprocals`.
"""
import numpy as np

__all__ = ['normalized_reciprocals', 'l2_loss', 'EigenvalueObjective', 'RatioObjective',
           'SpectralMatch']


def normalized_reciprocals(lams, jac=False):
    """`lam_1/lam_2, ..., lam_1/lam_K` -- the spectrum with its scale divided out.

    Scaling a domain by `c` divides every eigenvalue by `c^2`, so every ratio `lam_i/lam_j` is
    invariant and these `K-1` numbers are coordinates on the spectrum modulo size. Reciprocals of
    the FIRST eigenvalue rather than arbitrary pairs because `lam_1` is simple for any domain
    (Krahn), so the denominator is the one entry guaranteed not to be part of a cluster.

    They are also bounded: `lam_1 <= lam_j`, so every coordinate lies in `(0, 1]`, and a loss
    built on them weights the low modes and the high ones comparably. Matching raw eigenvalues
    would let `lam_K` dominate simply for being large.

    With `jac=True` also returns the `(K-1, K)` Jacobian.
    """
    lams = np.asarray(lams, dtype=float)
    if lams.ndim != 1 or len(lams) < 2:
        raise ValueError(f'need at least two eigenvalues, got shape {lams.shape}')
    if np.any(lams <= 0):
        raise ValueError(f'eigenvalues must be positive, got {lams}')
    nu = lams[0]/lams[1:]
    if not jac:
        return nu
    J = np.zeros((len(lams) - 1, len(lams)))
    J[:, 0] = 1.0/lams[1:]
    J[np.arange(len(lams) - 1), np.arange(1, len(lams))] = -lams[0]/lams[1:]**2
    return nu, J


def l2_loss(x, target, jac=False, log=False):
    """`||x - target||^2`, or its logarithm.

    `log=True` is worth having for a spectral match: the loss falls through many orders of
    magnitude as the shape is recovered, and the logarithm keeps the gradient's SCALE roughly
    constant over that range, which line searches tuned near the start still handle near the end.
    Its gradient is the plain one divided by the loss, so it is undefined at an exact match --
    the routine refuses rather than returning an infinity.
    """
    x = np.asarray(x, dtype=float)
    target = np.asarray(target, dtype=float)
    if x.shape != target.shape:
        raise ValueError(f'shape mismatch: x is {x.shape}, target is {target.shape}')
    diff = x - target
    val = float(diff @ diff)
    if log and val == 0.0:
        raise ValueError('log loss is undefined at an exact match')
    out = np.log(val) if log else val
    if not jac:
        return out
    grad = 2*diff
    return out, (grad/val if log else grad)


class EigenvalueObjective:
    """Minimize (or maximize) a single eigenvalue, by index.

    `sign=-1` maximizes. The perimeter constraint lives in the parametrization, so no penalty or
    multiplier is needed here.

    THE INDEX IS THE HAZARD. This tracks `lam_k` BY POSITION, and eigenvalues cross as the shape
    moves. Worse, the optima of these problems are typically AT a crossing -- the extremal shape
    is usually the one where `lam_k` meets its neighbour -- so an optimizer run to convergence
    tends to end up exactly where the index stops being meaningful and `lam` stops being
    differentiable. `douse.evaluate.SpectrumEvaluator` warns when the gap closes; heeding it is
    the caller's job, and handling it properly is the open design question in this package.
    """

    def __init__(self, k=0, sign=1.0):
        if k < 0:
            raise ValueError(f'k is a 0-based index into the spectrum, got {k}')
        self.k = int(k)
        self.sign = float(sign)

    def __repr__(self):
        what = 'minimize' if self.sign > 0 else 'maximize'
        return f'EigenvalueObjective({what} lam_{self.k + 1})'

    @property
    def n_eigs(self):
        """How many eigenvalues an evaluator must supply."""
        return self.k + 1

    def __call__(self, lams, dlams):
        """`(value, gradient)` from the spectrum and its `(K, n_p)` Jacobian."""
        lams = np.asarray(lams, dtype=float)
        if len(lams) <= self.k:
            raise ValueError(f'objective needs lam_{self.k + 1} but only {len(lams)} '
                             f'eigenvalues were supplied')
        return self.sign*float(lams[self.k]), self.sign*np.asarray(dlams)[self.k]


class SpectralMatch:
    """Match a target spectrum up to scale: the inverse problem.

    `targets` is a spectrum from the unknown domain -- absolute eigenvalues are fine, since only
    their ratios are used, so the target's size never has to be known or guessed.

    Composes three maps, each with its own Jacobian:

        p  ->  lam (K)  ->  nu = lam_1/lam_j (K-1)  ->  ||nu - nu_target||^2

    and the gradient is the product. `lam -> nu` is where the scale leaves the problem.
    """

    def __init__(self, targets, log=False):
        targets = np.asarray(targets, dtype=float)
        if len(targets) < 2:
            raise ValueError('a scale-invariant match needs at least two target eigenvalues; '
                             'with one there is nothing but the scale')
        self.targets = targets
        self.nu_target = normalized_reciprocals(targets)
        self.log = bool(log)

    def __repr__(self):
        return f'SpectralMatch({len(self.targets)} eigenvalues, log={self.log})'

    @property
    def n_eigs(self):
        return len(self.targets)

    def __call__(self, lams, dlams):
        lams = np.asarray(lams, dtype=float)
        if len(lams) != len(self.targets):
            raise ValueError(f'expected {len(self.targets)} eigenvalues, got {len(lams)}')
        nu, dnu_dlam = normalized_reciprocals(lams, jac=True)
        loss, dloss_dnu = l2_loss(nu, self.nu_target, jac=True, log=self.log)
        return loss, dloss_dnu @ dnu_dlam @ np.asarray(dlams)


class RatioObjective:
    """`lam_k / lam_1` -- the polygon analogue of Osting's spectral-ratio problems.

    `sign=-1` (the default) maximizes, since `descend` minimizes; the reported value is
    `sign * lam_k/lam_1`, so a maximization run reports a negative number that decreases.

    SCALE-INVARIANT BY CONSTRUCTION, which is why it fits this parametrization so well: scaling a
    domain divides every eigenvalue by the same factor, so the ratio does not see size at all.
    The fixed perimeter is then pure gauge -- it removes the scaling degree of freedom without
    imposing any physics on the objective.

    KNOWN STRUCTURE, for k = 2. Over all planar domains `lam_2/lam_1` is maximized by the disk at
    2.5387340 (Payne--Polya--Weinberger). Measured over regular N-gons it climbs monotonically
    towards that: 7/3 at N=3, exactly 5/2 at N=4, then 2.5268387, 2.5340068, 2.5365339, 2.5375926
    for N=5..8. And at EVERY regular N-gon `lam_2 = lam_3` exactly -- so if the regular N-gon is
    the maximizer, this objective's optimum sits on a degeneracy, where `lam_2` is only
    directionally differentiable and no gradient method converges.

    `douse.certify` handles that case without an optimizer: at the regular N-gon
    `grad(lam_1) = 0` as well, and local maximality reduces to a determinant condition on the
    2 x 2 cluster matrices.
    """

    def __init__(self, k=1, sign=-1.0):
        if k < 1:
            raise ValueError(f'k must be at least 1 (0-based); lam_1/lam_1 is constant. Got {k}')
        self.k = int(k)
        self.sign = float(sign)

    def __repr__(self):
        what = 'maximize' if self.sign < 0 else 'minimize'
        return f'RatioObjective({what} lam_{self.k + 1}/lam_1)'

    @property
    def n_eigs(self):
        return self.k + 1

    def __call__(self, lams, dlams):
        lams = np.asarray(lams, dtype=float)
        if len(lams) <= self.k:
            raise ValueError(f'objective needs lam_{self.k + 1} but only {len(lams)} '
                             f'eigenvalues were supplied')
        dlams = np.asarray(dlams, dtype=float)
        l1, lk = lams[0], lams[self.k]
        grad = (dlams[self.k]*l1 - lk*dlams[0])/l1**2
        return self.sign*float(lk/l1), self.sign*grad
