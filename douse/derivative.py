"""Shape derivatives of Laplacian eigenvalues with respect to a polygon's parameters.

    dlam/dp  =  (dlam/d vertices) . (d vertices/dp)

The right factor is pure geometry and lives in `douse.polygon`. The left factor is the Hadamard
formula, evaluated with `lappy`'s Cauchy-data machinery:

    dlam  =  - integral_{dOmega} (du/dn)^2 (V.n) ds

for a simple eigenvalue with `u` normalized in L^2. `lappy` implements no shape-derivative
formulas -- it provides `weighted_integral`, `normal_velocity` and the quadrature under them, and
this module is the formula.

ONE SOLVE BUYS THE WHOLE GRADIENT. The eigenfunction's Cauchy data is evaluated once; each of the
`2N` partial derivatives is then a weighted integral against a different velocity field, costing
a dot product. So `dlam_dp` is about as expensive as a single eigenvalue evaluation, not `2N` of
them -- which is the reason to use Hadamard rather than differencing the solver.

THE NODE SET IS NOT THE OBVIOUS ONE. Every vertex perturbation MOVES A CORNER, and a
corner-moving velocity is `V.n ~ r` at that corner, outside the exactness class of the node set
that `lappy` uses to normalize eigenfunctions. Use `solver.hadamard_quad`, never
`solver.bdry_quad`: on an L-shape's reentrant vertex the difference is 6.1 digits against 9.3,
with both answers looking entirely plausible. See lappy's `docs/polygon_contract.md`.
"""
import numpy as np

try:                    # the lappy Hadamard path, for the deprecated lappy backend (douse[lappy])
    from lappy.eigfun_integrals import eigfun_cauchy_data, normal_velocity, weighted_integral
except ImportError:
    def _needs_lappy(*_a, **_k):
        raise ImportError("this derivative path is the lappy backend's: install douse[lappy]")
    eigfun_cauchy_data = normal_velocity = weighted_integral = _needs_lappy

__all__ = ['vertex_velocity', 'velocity_fields', 'dlam_dvertices', 'dlam_dp',
           'dlams_dvertices', 'dlams_dp', 'detect_cluster', 'cluster_matrices',
           'directional_derivatives']


def vertex_velocity(bq, k, n_vert, direction=1.0 + 0.0j):
    """Per-node boundary displacement `dp` for moving vertex `k` by `direction`.

    Segment `j` of a `lappy.Polygon` runs from vertex `j` to vertex `j+1` (mod N), parametrized
    by normalized arclength `tau` in [0, 1]. Moving vertex `k` therefore displaces

        the END of segment k-1   by  tau * direction
        the START of segment k   by  (1 - tau) * direction

    and leaves every other node fixed: the piecewise-linear vertex perturbation, the standard
    velocity field for a polygon parametrized by its vertices. Both edges stay straight, so this
    is exact rather than a linearization.

    Uses `bq.seg_idx` and `bq.tau`, the per-node provenance `lappy` carries for exactly this.
    """
    if bq.seg_idx is None or bq.tau is None:
        raise ValueError(
            'this BoundaryQuad carries no per-node seg_idx/tau, so a velocity field cannot be '
            'evaluated on it. It predates lappy 0.9 -- rebuild the solver.')
    dp = np.zeros(len(bq.pts), dtype=complex)
    on_prev = bq.seg_idx == (k - 1) % n_vert
    on_next = bq.seg_idx == k % n_vert
    dp[on_prev] = bq.tau[on_prev]*direction
    dp[on_next] = (1.0 - bq.tau[on_next])*direction
    return dp


def velocity_fields(bq, n_vert):
    """`V.n` at every node for each of the `2N` vertex coordinate directions.

    Shape `(2N, n_nodes)`, ordered `[d/dx_0, ..., d/dx_{N-1}, d/dy_0, ..., d/dy_{N-1}]`.

    These depend on the GEOMETRY ALONE -- not on lambda, not on the eigenfunction -- so a
    K-eigenvalue Jacobian builds them once and reuses them for every mode. That is most of why
    `dlams_dp` costs barely more than `dlam_dp`.
    """
    fields = np.empty((2*n_vert, len(bq.pts)))
    for k in range(n_vert):
        for axis, direction in ((0, 1.0 + 0.0j), (1, 0.0 + 1.0j)):
            fields[axis*n_vert + k] = normal_velocity(bq, vertex_velocity(bq, k, n_vert,
                                                                          direction))
    return fields


def _refuse_multiple(mult):
    if mult != 1:
        raise NotImplementedError(
            f'mult={mult}: the shape derivative of a multiple eigenvalue is not a gradient. It '
            f'is the set of eigenvalues of the m x m matrix `-int (du_i/dn)(du_j/dn) (V.n) ds`, '
            f'one matrix per direction, and lam is only directionally differentiable there. '
            f'`lappy.weighted_integral` already returns that matrix for mult>1, so the pieces '
            f'exist; what is missing is a decision about which optimizer consumes them. '
            f'Degenerate optima are the rule for these problems, so this is worth doing properly '
            f'rather than papering over.')


def dlams_dvertices(solver, lams, n_vert, bq=None):
    """Jacobian of several SIMPLE eigenvalues w.r.t. the vertex coordinates: `(K, 2N)`, real.

    One row per eigenvalue, same column layout as `dlam_dvertices`. The velocity fields are
    built once and shared, so the marginal cost of another mode is one `eigenfunction_coef` plus
    one Cauchy-data evaluation -- not another `2N` integrals from scratch.
    """
    if hasattr(solver, 'chladni_dlams_dvertices'):       # the chladni backend: its own rule
        return solver.chladni_dlams_dvertices(lams, n_vert)
    if bq is None:
        bq = solver.hadamard_quad
    fields = velocity_fields(bq, n_vert)
    J = np.empty((len(lams), 2*n_vert))
    for j, lam in enumerate(lams):
        coef = solver.eigenfunction_coef(float(lam), mult=1)     # L^2-orthonormal
        ed = eigfun_cauchy_data(solver.basis, float(lam), coef, bq)
        for i in range(2*n_vert):
            J[j, i] = -weighted_integral(ed, 'NN', fields[i])[0, 0]
    return J


def dlams_dp(family, p, solver, lams, bq=None):
    """Jacobian of several eigenvalues w.r.t. the parameters: `(K, 2N-4)`, real.

    The chain rule applied row-wise. This is what an objective built on more than one eigenvalue
    -- a spectral match, an eigenvalue gap -- differentiates through.
    """
    return dlams_dvertices(solver, lams, family.n, bq=bq) @ family.d_vertices_dp(p)


def dlam_dvertices(solver, lam, n_vert, mult=1, bq=None):
    """Gradient of `lam` with respect to the vertex coordinates: shape `(2N,)`, real.

    Layout is `[d/dx_0, ..., d/dx_{N-1}, d/dy_0, ..., d/dy_{N-1}]`, matching
    `douse.polygon.d_vertices_dp`'s row order.

    These are UNCONSTRAINED partials: each moves one vertex and lets the perimeter change. The
    perimeter constraint enters one level up, through the parametrization's Jacobian.

    `bq` defaults to `solver.hadamard_quad`, which is the correct node set here and not the one
    `solver.bdry_quad` gives (see the module docstring). Pass one explicitly only if you know why.
    """
    _refuse_multiple(mult)
    return dlams_dvertices(solver, [lam], n_vert, bq=bq)[0]


def dlam_dp(family, p, solver, lam, mult=1, bq=None):
    """Gradient of `lam` with respect to the parameter vector: shape `(2N-4,)`, real.

    The chain rule, and the whole point of the module:

        dlam/dp = dlam/d(vertices) @ d(vertices)/dp

    `solver` must be a solver for the domain `family(p)` and `lam` one of its eigenvalues; this
    routine does not solve, so that a caller running an optimization can reuse the solver it
    already built.
    """
    _refuse_multiple(mult)
    return dlams_dp(family, p, solver, [lam], bq=bq)[0]


# ── Degenerate clusters ──────────────────────────────────────────────────────────
#
# Where `dlams_dp` stops applying. At a multiple eigenvalue the modes that emerge under a
# perturbation `V` are the eigenvalues of an m x m matrix linear in `V`, and the routines below
# expose that matrix rather than pretending a gradient exists.


def detect_cluster(lams, k, rtol=1e-6):
    """`(lo, hi)` bounding the cluster of `lams` that contains index `k`, as a half-open range.

    Membership is decided by RELATIVE GAP in the computed spectrum, with the tolerance in the
    caller's hands. It is deliberately not delegated to the solver's own multiplicity estimate:
    that quantity is least reliable exactly here, and `lappy`'s got it wrong in both directions
    on `reg_ngon_6` -- reporting a double at a simple eigenvalue, while its reference table
    listed a genuine double as simple.
    """
    lams = np.asarray(lams, dtype=float)
    if not 0 <= k < len(lams):
        raise IndexError(f'k={k} is outside a spectrum of length {len(lams)}')
    order = np.argsort(lams)
    pos = int(np.where(order == k)[0][0]) if not np.all(np.diff(lams) >= 0) else k
    srt = lams[order]
    lo = hi = pos
    while lo > 0 and (srt[lo] - srt[lo - 1]) <= rtol*abs(srt[lo]):
        lo -= 1
    while hi + 1 < len(srt) and (srt[hi + 1] - srt[hi]) <= rtol*abs(srt[hi]):
        hi += 1
    return lo, hi + 1


def cluster_matrices(family, p, solver, lam, mult, bq=None, lams=None):
    """`(n_p, m, m)` -- the matrix `M^(i)` for each parameter direction.

    `lams`, if given, is the m individual eigenvalues of the cluster, and the basis is then built
    from their OWN eigenfunctions (`mult=1` each) rather than from the m-dimensional cluster.
    That matters whenever the cluster is only NEARLY degenerate and a caller wants to add the
    actual splitting `Lambda = diag(lam_i - lam_k)` to the model: `Lambda` is diagonal only in
    the basis that diagonalizes the unperturbed operator, and `eigenfunction_coef(mult=m)`
    returns an ARBITRARY Loewdin basis of the eigenspace instead. Adding the two in mismatched
    bases produces a model whose predicted direction is wrong -- measured as a trust-region ratio
    that stays negative (-0.2 to -1.2) as the radius shrinks, where a correct first-order model
    gives +1.

    At an exact degeneracy the individual eigenfunctions are not defined, but `Lambda = 0` there
    and any orthonormal basis of the eigenspace serves, so pass `lams=None`.

    `M(d) = sum_i d_i M^(i)` is linear in `d`, and its EIGENVALUES are the one-sided directional
    derivatives of the m eigenvalues in the cluster. That linearity is the whole reason this
    problem is tractable: the local model of an ordered eigenvalue is an eigenvalue of an affine
    matrix function, which is the well-understood case.

    Only the spectrum of `M(d)` means anything. `eigenfunction_coef(mult=m)` returns an arbitrary
    orthonormal basis of the eigenspace, so the entries depend on a choice that moves under
    perturbation -- never difference them across iterates, never cache the basis.
    """
    if hasattr(solver, 'chladni_cluster_matrices'):      # the chladni backend
        return solver.chladni_cluster_matrices(family, p, lam, mult, lams)
    if bq is None:
        bq = solver.hadamard_quad
    fields = velocity_fields(bq, family.n)
    Jv = family.d_vertices_dp(p)
    if lams is None:
        coef = solver.eigenfunction_coef(float(lam), mult=int(mult))
    else:
        coef = np.hstack([solver.eigenfunction_coef(float(l), mult=1) for l in lams])
    ed = eigfun_cauchy_data(solver.basis, float(lam), coef, bq)
    return np.array([-weighted_integral(ed, 'NN', fields.T @ Jv[:, i])
                     for i in range(Jv.shape[1])])


def directional_derivatives(Ms, d):
    """Sorted directional derivatives of the cluster in direction `d`: `eig(sum_i d_i M^(i))`."""
    d = np.asarray(d, dtype=float)
    return np.linalg.eigvalsh(np.einsum('i,ijk->jk', d, np.asarray(Ms)))
