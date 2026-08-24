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

from lappy.eigfun_integrals import eigfun_cauchy_data, normal_velocity, weighted_integral

__all__ = ['vertex_velocity', 'velocity_fields', 'dlam_dvertices', 'dlam_dp',
           'dlams_dvertices', 'dlams_dp']


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
