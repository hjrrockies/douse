"""Polygons parametrized by their vertices, with the perimeter held fixed.

THE PARAMETRIZATION. An N-gon in the plane has 2N degrees of freedom, but a Laplacian
eigenvalue does not see all of them: translating, rotating or reflecting the domain leaves the
spectrum alone, and scaling it by `c` divides every eigenvalue by `c^2`. Optimizing over all 2N
would send the optimizer wandering a four-dimensional flat manifold of identical spectra. So the
gauge is fixed by construction:

    vertices[N-1] = 0                 translation fixed: the last vertex is the origin
    vertices[0]   real and positive   rotation fixed: the closing edge is on the +x axis
    perimeter     = `perim`           scaling fixed

leaving `vertices[1..N-2]` free -- N-2 complex numbers, so **2N-4 real parameters**

    p = [x_1, ..., x_{N-2}, y_1, ..., y_{N-2}]

and `vertices[0]` is then determined by the perimeter constraint. Reflection is not fixed here;
it is handled by requiring counter-clockwise orientation, which `lappy.Polygon` enforces.

WHICH DIRECTION IS SAFE. `polygon_vertices` (p -> vertices) is smooth, and it is the only map an
optimizer should call. `param_vector` (vertices -> p) is NOT smooth: it picks the longest side
and conditionally reverses orientation, and both choices jump when two sides are nearly equal.
That matters more than it sounds, because the shapes these problems converge to are the regular
ones, where ALL sides are equal -- the canonical form is most degenerate exactly at the expected
answer. Use `param_vector` to build an initial guess and never again inside a loop.
"""
import numpy as np


# Geometry helpers, lappy's conventions exactly (`lappy/utils.py`), so douse no longer imports
# lappy for them (2026-10-14).
def _complex_form(pts):
    pts = np.asarray(pts)
    if np.iscomplexobj(pts):
        return pts
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts[:, 0] + 1j*pts[:, 1]
    return pts.astype(complex)


def edge_lengths(vertices):
    """Edge lengths, edge i running from vertex i to vertex i+1."""
    v = _complex_form(vertices)
    return np.abs(np.roll(v, -1) - v)


def polygon_area(vertices):
    """Signed area by the shoelace formula: positive for counter-clockwise vertices."""
    v = _complex_form(vertices)
    x, y = v.real, v.imag
    return 0.5*np.sum((x - np.roll(x, -1))*(y + np.roll(y, -1)))


def polygon_perimeter(vertices):
    return float(np.sum(edge_lengths(vertices)))

__all__ = ['polygon_vertices', 'd_vertices_dp', 'param_vector', 'gauge_vector',
           'perimeter_gradient', 'VertexPolygon', 'n_params', 'n_vertices']


def n_params(n_vert):
    """Number of real parameters for an `n_vert`-gon."""
    return 2*n_vert - 4


def n_vertices(p):
    """Number of vertices the parameter vector `p` describes."""
    p = np.asarray(p)
    if p.ndim != 1 or len(p) < 2 or len(p) % 2:
        raise ValueError(f'p must be a 1-D even-length array of length >= 2, got shape {p.shape}')
    return len(p)//2 + 2


def _split(p):
    """`(x, y, N)` for the free vertices."""
    N = n_vertices(p)
    p = np.asarray(p, dtype=float)
    return p[:N-2], p[N-2:], N


def _closing_budget(x, y, perim):
    """Arclength left for the two edges that close the polygon through `vertices[0]`.

    The free vertices `v_1..v_{N-2}` fix the chain `v_1 -> v_2 -> ... -> v_{N-2} -> 0`; whatever
    perimeter remains has to be spent on `0 -> v_0` and `v_0 -> v_1`.
    """
    chain = np.concatenate([x + 1j*y, [0.0 + 0.0j]])       # v_1..v_{N-2}, then the origin
    return perim - np.abs(np.diff(chain)).sum()


def polygon_vertices(p, perim=1.0):
    """Vertices (complex, length N) of the polygon described by `p`.

    `vertices[0]` solves the perimeter constraint. Writing `a = vertices[0]` (real), `C` for the
    remaining budget and `v_1 = x_1 + i y_1`, the constraint `|a - v_1| + a = C` gives

        a = (C^2 - x_1^2 - y_1^2) / (2 (C - x_1))

    which is real and positive whenever `C > |v_1|`, i.e. whenever the two closing edges can
    actually reach around. Raises otherwise, rather than returning a complex or negative `a`
    that would silently produce a nonsense polygon.
    """
    x, y, N = _split(p)
    C = _closing_budget(x, y, perim)
    if C <= 0:
        raise ValueError(
            f'the free vertices already use {perim - C:.6g} of the {perim:.6g} perimeter budget, '
            f'leaving {C:.3g} for the two closing edges: no such polygon exists')
    r1 = np.hypot(x[0], y[0])
    if C <= r1:
        raise ValueError(
            f'the closing budget C={C:.6g} is not more than |v_1|={r1:.6g}, so the path '
            f'0 -> v_0 -> v_1 cannot reach: no such polygon exists')

    vertices = np.zeros(N, dtype=complex)
    vertices[1:-1] = x + 1j*y
    vertices[0] = 0.5*(C*C - x[0]**2 - y[0]**2)/(C - x[0])
    return vertices


def perimeter_gradient(vertices):
    """`(perimeter, grad)` where `grad[k]` is `dP/dx_k + i dP/dy_k`.

    `P = sum_k |v_k - v_{k-1}|`, so with `u_k` the unit vector along edge `k`,
    `dP/dv_k = u_k - u_{k+1}`.
    """
    dv = vertices - np.roll(vertices, 1)
    lengths = np.abs(dv)
    if np.any(lengths == 0):
        raise ValueError('polygon has a zero-length edge; the perimeter gradient is undefined')
    u = dv/lengths
    return lengths.sum(), u - np.roll(u, -1)


def d_vertices_dp(p, perim=1.0):
    """Jacobian of the vertices with respect to `p`: shape `(2N, 2N-4)`, real.

    Row order is `[x_0, ..., x_{N-1}, y_0, ..., y_{N-1}]`, matching what
    `douse.derivative.dlam_dvertices` returns, so the chain rule is a plain matrix product.

    Only one row is interesting. `x_j` and `y_j` for `j = 1..N-2` ARE the parameters, so those
    rows are identity blocks; `v_{N-1}` is pinned at the origin and `y_0` at zero, so those rows
    vanish. `x_0` is determined by the perimeter constraint, and its row comes from
    differentiating `P(v) = perim` implicitly:

        dP/dx_0 * dx_0/dp  +  sum_j [dP/dx_j * dx_j/dp + dP/dy_j * dy_j/dp]  =  0

    IMPLICIT, NOT EXPLICIT. Solving the quadratic for `x_0` and differentiating it directly is
    possible and was tried in an earlier version of this package, where it was left commented out
    and marked BUGGED. The implicit form needs only the perimeter gradient, which is three lines
    and hard to get wrong, and it is what `test_polygon.py` verifies against a finite difference.
    """
    x, y, N = _split(p)
    vertices = polygon_vertices(p, perim)
    _, grad = perimeter_gradient(vertices)

    dP_dx0 = grad[0].real
    if dP_dx0 == 0:
        raise ValueError('dP/dx_0 vanishes: the perimeter constraint does not determine '
                         'vertices[0] here, so the parametrization is degenerate at this p')

    m = len(p)//2                                          # number of free vertices, N-2
    J = np.zeros((2*N, len(p)))
    # x_0: implicit
    J[0, :m] = -grad[1:-1].real/dP_dx0
    J[0, m:] = -grad[1:-1].imag/dP_dx0
    # x_j, y_j for j = 1..N-2: the parameters themselves
    J[1:N-1, :m] = np.eye(m)
    J[N+1:2*N-1, m:] = np.eye(m)
    # y_0 (row N) and v_{N-1} (rows N-1, 2N-1) are pinned: left as zeros
    return J


def param_vector(vertices, perim=1.0):
    """The canonical parameter vector for a polygon -- FOR INITIALIZATION ONLY.

    Puts the polygon in the gauge `polygon_vertices` assumes: longest side last, orientation
    normalized, last vertex at the origin, closing edge along +x, perimeter `perim`.

    NOT SMOOTH, and its discontinuities sit where you are going. `argmax` over the side lengths
    jumps when two sides tie, and the orientation flip jumps when the two sides adjacent to the
    longest tie -- and a regular polygon, the answer to most of these optimization problems, ties
    everywhere. Calling this inside an optimizer puts a branch cut through the target. Use it
    once, to turn a shape you have into a starting `p`, and then work with `p`.
    """
    vertices = np.asarray(vertices, dtype=complex).copy()

    # Orientation first, and by AREA rather than by side lengths. An earlier version reversed
    # when `lens[-2] > lens[0]`, a tie-break about which end of the longest side to start from --
    # but reversing flips the orientation, and `lappy.Polygon` requires counter-clockwise, so on
    # a CCW input that rule produced a polygon outside the parametrization's own range. With CCW
    # imposed and the longest side placed last there is no freedom left to break.
    if polygon_area(vertices) < 0:
        vertices = vertices[::-1]

    # longest side last, so it becomes the closing edge that the gauge puts on the +x axis
    vertices = np.roll(vertices, -int(np.argmax(edge_lengths(vertices))) - 1)

    vertices = vertices - vertices[-1]
    vertices = vertices*np.exp(-1j*np.angle(vertices[0] - vertices[-1]))
    vertices = vertices/(edge_lengths(vertices).sum()/perim)
    return np.concatenate([vertices[1:-1].real, vertices[1:-1].imag])


def gauge_vector(vertices, perim=1.0, anchor=0):
    """`param_vector` WITHOUT the longest-side roll: the caller chooses which vertex anchors.

    `param_vector` fixes the gauge and also picks the labelling, by putting the longest side last.
    That second choice is the non-smooth one, and it is also the wrong one when a caller has
    already decided which vertex should be `v_0` -- it silently overrides them.

    IT HAS TO BE OVERRIDABLE, because `v_0` is the vertex `polygon_vertices` solves for and the
    solve fails when `v_0` is FLAT. The two closing edges are then collinear, so the closing
    budget `C` equals `|v_1|` exactly and the constructor refuses with "no such polygon exists" --
    on a perfectly good shape, purely because of how it was labelled. Measured on the unit square
    written as an 8-gon with four flat vertices, which is exactly the representation
    `douse.constraints` relies on being available: `param_vector` rolls a flat vertex into `v_0`
    (every side ties at 0.5, so `argmax` takes the first) and the round trip raises.

    `anchor` is the index that becomes `v_0`. `douse.degeneracy.canonicalize_p` passes the corner
    furthest from flat.
    """
    vertices = np.asarray(vertices, dtype=complex).copy()
    if polygon_area(vertices) < 0:
        vertices = vertices[::-1]
    vertices = np.roll(vertices, -int(anchor))
    vertices = vertices - vertices[-1]
    vertices = vertices*np.exp(-1j*np.angle(vertices[0] - vertices[-1]))
    vertices = vertices/(edge_lengths(vertices).sum()/perim)
    return np.concatenate([vertices[1:-1].real, vertices[1:-1].imag])


class VertexPolygon:
    """A family of `n`-gons of fixed perimeter, parametrized by `2n-4` real numbers.

    This is the `ParametricDomain` that `lappy`'s scope document leaves to the downstream
    package: the map from a parameter vector to a `lappy.Domain`, plus its derivative. `lappy`
    owns *a* domain and its conventions; this owns *families* of them.

        fam = VertexPolygon(4)
        dom = fam(p)                 # a chladni Polygon (Dirichlet)
        J   = fam.d_vertices_dp(p)   # (2n, 2n-4)
    """

    def __init__(self, n, perim=1.0):
        if n < 3:
            raise ValueError(f'a polygon needs at least 3 vertices, got n={n}')
        if perim <= 0:
            raise ValueError(f'perim must be positive, got {perim}')
        self.n = int(n)
        self.perim = float(perim)

    def __repr__(self):
        return f'VertexPolygon(n={self.n}, perim={self.perim:g}, n_params={self.n_params})'

    @property
    def n_params(self):
        return n_params(self.n)

    def _check(self, p):
        p = np.asarray(p, dtype=float)
        if len(p) != self.n_params:
            raise ValueError(f'p must have length {self.n_params} for n={self.n}, got {len(p)}')
        return p

    def vertices(self, p):
        return polygon_vertices(self._check(p), self.perim)

    def d_vertices_dp(self, p):
        return d_vertices_dp(self._check(p), self.perim)

    def __call__(self, p, **kwargs):
        """The `chladni.geometry.Polygon` for `p`, Dirichlet unless `bc` is given (chladni is
        douse's backend since 2026-10-14; `lappy_domain` gives the deprecated lappy one).

        Orientation is checked here rather than left to the solver, because the message matters to
        an optimizer: a counter-clockwise polygon and its mirror image are both valid shapes with
        the same spectrum, but only one of them is in this parametrization's range.
        """
        vertices = self.vertices(p)
        if polygon_area(vertices) < 0:
            raise ValueError(
                'this p gives a clockwise polygon, which is outside the parametrization\'s '
                'range. Its mirror image has the same spectrum and is inside; `param_vector` '
                'produces the right orientation from any simple polygon.')
        from chladni.geometry import Polygon
        kwargs.setdefault("bc", "dir")
        return Polygon(vertices, **kwargs)

    def lappy_domain(self, p, **kwargs):
        """The `lappy.Polygon` for `p`. DEPRECATED: for the lappy backend only (douse[lappy])."""
        from lappy import Polygon
        vertices = self.vertices(p)
        if polygon_area(vertices) < 0:
            raise ValueError('this p gives a clockwise polygon, which is outside the '
                             'parametrization\'s range')
        return Polygon(vertices, **kwargs)

    def is_valid(self, p):
        """Does `p` describe a simple, correctly-oriented polygon? Never raises.

        A cheap admissibility test for an optimizer that has stepped somewhere strange. It is a
        yes/no, not a constraint function -- a smooth self-intersection penalty is a separate
        thing and is not implemented yet.
        """
        try:
            self(p)
            return True
        except (ValueError, TypeError):
            return False

    def perimeter(self, p):
        """Measured, not assumed -- this is the constraint `polygon_vertices` claims to enforce
        and it costs nothing to check."""
        return polygon_perimeter(self.vertices(p))


def mirror_axes(vertices):
    """Every candidate mirror line of a polygon, best first: `[(defect, s, a), ...]`.

    A reflection that maps the polygon onto itself sends vertex `i` to vertex `s - i (mod N)` for
    one shift `s`, and about the centroid it is `z -> a conj(z)` with `|a| = 1`; the mirror line
    is at angle `arg(a)/2`. For each `s` the best `a` is the phase of `sum w_i v_i` (`w` the
    relabelled vertices), and `defect` is the RMS distance between the reflected polygon and
    the relabelled one, divided by the perimeter: zero for an exact mirror symmetry.
    """
    v = np.asarray(vertices, dtype=complex)
    v = v - v.mean()
    N, P = len(v), polygon_perimeter(v)
    out = []
    for s in range(N):
        w = v[(s - np.arange(N)) % N]
        a = np.exp(1j*np.angle(np.sum(w*v)))
        out.append((float(np.sqrt(np.mean(np.abs(a*np.conj(v) - w)**2))/P), s, a))
    return sorted(out, key=lambda t: t[0])


class MirrorPolygon:
    """`n`-gons with a mirror symmetry, of fixed perimeter: `n - 2` real parameters.

    WHY. Several `max lam_k/lam_1` optima look symmetric to about 2e-3 of the perimeter but not
    to the 1e-4 that independent starts agree to. Over all N-gons a symmetry is not something
    the optimizer can find exactly: the derivatives that would restore it are at the noise floor
    of the shape derivative. Restricting to symmetric polygons makes it exact, halves the
    parameter count, and leaves the question of whether the symmetric optimum is a maximum among
    ALL N-gons to a separate test along the antisymmetric directions
    (`antisymmetric_basis`, `douse.certify.antisymmetric_hessian`).

    THE LAYOUT. The mirror line is the y-axis and the vertices run counter-clockwise; a vertex
    `P` on the right (x > 0) has its image `-conj(P)` on the left. `axis` says what the line
    passes through:

        'vertex' (odd n)    the bottom vertex B = 0, pairs P_1..P_m (m = (n-1)/2), bottom to top
        'vertex' (even n)   B = 0, pairs P_1..P_m (m = (n-2)/2), the top vertex T = i t
        'edge'   (even n)   pairs P_1..P_m (m = n/2): the axis crosses the edges M(P_1)P_1 and
                            P_m M(P_m)

    and the vertex order is `[B,] P_1, .., P_m, [T,] M(P_m), .., M(P_1)`. The gauge: B at the
    origin (or P_1 on the x-axis for 'edge'), and x_1 = 1 for scale, after which the polygon is
    rescaled to perimeter `perim`. The parameters are y_1 (not for 'edge'), then x_j, y_j for
    j = 2..m, then t (even 'vertex' only).

    THE DEGENERACY MOVES OF `douse.cluster_opt` DO NOT APPLY: they propose `VertexPolygon`
    parameters, which `is_valid` rejects here by length. A symmetric run cannot flatten a
    vertex or collapse an edge.
    """

    def __init__(self, n, axis='vertex', perim=1.0):
        if n < 3:
            raise ValueError(f'a polygon needs at least 3 vertices, got n={n}')
        if axis not in ('vertex', 'edge'):
            raise ValueError(f"axis must be 'vertex' or 'edge', got {axis!r}")
        if axis == 'edge' and n % 2:
            raise ValueError(f'an odd polygon\'s mirror line passes through a vertex; '
                             f"axis='edge' needs even n, got n={n}")
        self.n, self.axis, self.perim = int(n), axis, float(perim)
        self.m = (n - 1)//2 if n % 2 else (n - 2)//2 if axis == 'vertex' else n//2
        self._W = np.array([self._raw(e) - self._raw(np.zeros(self.n_params))
                            for e in np.eye(self.n_params)]).T          # (n, n_params), complex

    def __repr__(self):
        return f'MirrorPolygon(n={self.n}, axis={self.axis!r}, perim={self.perim:g})'

    @property
    def n_params(self):
        return self.n - 2

    @property
    def has_bottom(self):
        return self.axis == 'vertex'

    @property
    def has_top(self):
        return self.axis == 'vertex' and self.n % 2 == 0

    def _check(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_params,):
            raise ValueError(f'q must have length {self.n_params} for {self!r}, got {q.shape}')
        return q

    def _raw(self, q):
        """The unscaled vertices: affine in `q`."""
        q = list(q)
        P = [1.0 + 1j*(q.pop(0) if self.has_bottom else 0.0)]
        for _ in range(self.m - 1):
            x = q.pop(0)
            P.append(x + 1j*q.pop(0))
        top = [1j*q.pop(0)] if self.has_top else []
        bottom = [0j] if self.has_bottom else []
        P = np.array(P)
        return np.concatenate([bottom, P, top, -np.conj(P[::-1])])

    def mirror_index(self):
        """`sigma` with vertex `i` the image of vertex `sigma[i]`."""
        n = self.n
        return (-np.arange(n)) % n if self.has_bottom else (n - 1 - np.arange(n)) % n

    def vertices(self, q):
        w = self._raw(self._check(q))
        return self.perim*w/polygon_perimeter(w)

    def d_vertices_dp(self, q):
        """`(2n, n - 2)`, rows `[x_0..x_{n-1}, y_0..y_{n-1}]` as for `VertexPolygon`. The raw
        vertices are affine in `q`; the rescaling `v = perim w / P(w)` is differentiated
        exactly."""
        w = self._raw(self._check(q))
        P, g = perimeter_gradient(w)
        dP = (np.conj(g)[:, None]*self._W).real.sum(axis=0)              # (n_params,)
        dv = self.perim*(self._W/P - np.outer(w, dP)/P**2)
        return np.vstack([dv.real, dv.imag])

    def __call__(self, q, **kwargs):
        v = self.vertices(q)
        if polygon_area(v) <= 0:
            raise ValueError('this q gives a clockwise or degenerate polygon')
        from chladni.geometry import Polygon
        kwargs.setdefault('bc', 'dir')
        return Polygon(v, **kwargs)

    def lappy_domain(self, q, **kwargs):
        from lappy import Polygon
        return Polygon(self.vertices(q), **kwargs)

    def is_valid(self, q):
        try:
            self(q)
            return True
        except (ValueError, TypeError):
            return False

    def perimeter(self, q):
        return polygon_perimeter(self.vertices(q))

    @classmethod
    def from_vertices(cls, vertices, perim=1.0, rank=0):
        """`(family, q, defect)`: the symmetric polygon nearest `vertices` about its `rank`-th
        best mirror line (`mirror_axes`), in this layout. `defect` is that line's, so a caller
        can see how far the input was from symmetric."""
        v = np.asarray(vertices, dtype=complex)
        if polygon_area(v) < 0:
            v = v[::-1]
        N = len(v)
        defect, s, a = mirror_axes(v)[rank]
        u = (v - v.mean())*np.exp(1j*(np.pi/2 - np.angle(a)/2))     # the line is now the y-axis
        sig = (s - np.arange(N)) % N
        u = 0.5*(u - np.conj(u[sig]))                                  # exactly symmetric
        fixed = [i for i in range(N) if sig[i] == i]
        if fixed:                                  # the line passes through a vertex
            i0 = min(fixed, key=lambda i: u[i].imag)
            if len(fixed) == 1 and u[i0].imag > np.mean(u.imag):
                u = -u                             # a half-turn keeps the line and the orientation
            i0 = min(fixed, key=lambda i: u[i].imag)
            u = np.roll(u, -i0)
            fam = cls(N, 'vertex', perim)
            u = u - 1j*u[0].imag
        else:                                      # the line crosses two edges
            ends = [i for i in range(N) if sig[i] == (i + 1) % N]
            j = min(ends, key=lambda i: (u[i] + u[(i + 1) % N]).imag)
            u = np.roll(u, -((j + 1) % N))         # u[0] = P_1, right end of the bottom edge
            if u[0].real < 0:
                u = -u
            fam = cls(N, 'edge', perim)
            u = u - 1j*u[0].imag
        i1 = 1 if fam.has_bottom else 0
        if not u[i1].real > 0:
            raise ValueError('symmetrized polygon is not in the layout (P_1 not right of the axis)')
        u = u/u[i1].real
        q = [u[i1].imag] if fam.has_bottom else []
        for j in range(1, fam.m):
            q += [u[i1 + j].real, u[i1 + j].imag]
        if fam.has_top:
            q.append(u[fam.m + 1].imag)
        q = np.array(q)
        if not np.allclose(fam._raw(q), u, atol=1e-9):
            raise AssertionError('from_vertices: the layout does not reproduce the polygon')
        return fam, q, defect

    def antisymmetric_basis(self, q):
        """An orthonormal basis `(2n, n - 2)` of the vertex perturbations that BREAK the symmetry,
        rows `[x.., y..]`, with the two rigid motions that are antisymmetric (horizontal
        translation, rotation) removed. Along these `F(v + t d) = F(v - t d)` exactly (the
        mirror image is congruent), so first order vanishes and the curvature decides."""
        v = self.vertices(q)
        n, sig = self.n, self.mirror_index()
        cols = []
        for i in range(n):
            for e in (1.0, 1j):
                d = np.zeros(n, dtype=complex)
                d[i] = e
                cols.append(0.5*(d + np.conj(d[sig])))                 # minus its mirror image
        A = np.array([np.concatenate([c.real, c.imag]) for c in cols]).T
        rigid = np.array([np.concatenate([np.ones(n), np.zeros(n)]),
                          np.concatenate([(1j*v).real, (1j*v).imag])]).T
        Qr, _ = np.linalg.qr(rigid)
        A = A - Qr @ (Qr.T @ A)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        r = int(np.sum(S > 1e-10*S[0]))
        if r != n - 2:
            raise AssertionError(f'antisymmetric subspace has rank {r}, expected {n - 2}')
        return U[:, :r]
