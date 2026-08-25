"""Geometric constraints on a polygon family, as `c(p) >= 0` with a Jacobian.

WHY A MINIMUM EDGE LENGTH IS A GAUGE FIXING AND NOT A RESTRICTION ON SHAPES. An N-gon can lose a
vertex two ways, and they reach the SAME limiting domain:

    (A) edge collapse    |v_j - v_{j-1}| -> 0, two vertices merge
    (B) vertex flattening  interior angle -> pi, the vertex slides onto a straight edge

The spectrum sees only the domain, so in the limit these are indistinguishable. The
parametrization is another matter. Under (A) the two vertices are separately meaningful but the
domain depends on them, to O(eps), only through their common location -- so the difference
direction has a vanishing effect on `lam` and the Jacobian acquires a singular value going to
zero. That is a narrow curved valley, and it is what a trust-region method grinds against:
measured, `max lam_3/lam_1` at N=5 ran out of iterations with an edge at 1.0e-08 and a second
start ended with one at 0.042 and a 152 degree angle. Under (B) the slide direction is EXACTLY
null -- a tangential velocity has `V.n = 0`, so `dlam = 0` for every eigenvalue -- and the
objective is exactly constant along it rather than nearly constant. A flat valley an optimizer
ignores, instead of a narrow one it fights.

And (B) costs almost nothing to solve. The unit square written with flat vertices inserted, all
through the auto-configured path:

    4-gon, none          88 columns   12.0 digits
    5-gon, 1 flat        93 columns   10.3 digits
    6-gon, 2 flat       101 columns   11.6 digits
    8-gon, 4 flat       115 columns   10.8 digits

So: forbid (A), allow (B). Every M-gon with M < N is still reachable inside the N-gon family --
write it with N-M flat vertices inserted along its longest edges and every edge stays far above
the floor. The set of DOMAINS the optimizer can reach is unchanged; only the coordinates
describing them are pinned. An ANGLE floor would be a real restriction, because it forbids
degeneration outright; an EDGE floor forbids one of two representations of the same limit.

`l_min = perim/(10*n)` never binds on a shape anyone wants: it is a tenth of the average edge.

THE ONE THING THIS DOES NOT BUY. A short edge and a flat vertex describe the same domain but sit
in completely different regions of `p`-space, and a local method cannot tunnel between them. An
optimizer that wants an edge gone will press against the floor and stop, with the domain it
wanted freely available elsewhere in the family and no path to it. Converting a short edge into
a flat vertex is not domain-preserving at any usable threshold -- deleting an edge of length eps
moves `lam` by O(eps), which for any eps large enough to matter is far above solver accuracy. So
the constraint stops the damage and makes degeneration VISIBLE (see `douse.degeneracy`), but
reading the result still means re-solving in the reduced family. Redistributing vertices that
are ALREADY flat is exact, and `degeneracy.canonicalize` does that.
"""
import numpy as np

__all__ = ['Constraint', 'ConstraintSet', 'MinEdgeLength', 'ClosingBudget']


class Constraint:
    """`p -> (c, dc_dp)` with `c >= 0` admissible.

    Subclasses implement `_on_vertices(vertices)` returning `(c, dc_dv)` where `dc_dv` has shape
    `(m_c, 2N)` in the row order `[x_0..x_{N-1}, y_0..y_{N-1}]` -- the same order
    `VertexPolygon.d_vertices_dp` produces, so the chain rule is one matrix product. A constraint
    that is natural in `p` instead can override `__call__`.
    """

    def __init__(self, family):
        self.family = family

    @property
    def size(self):
        raise NotImplementedError

    def _on_vertices(self, vertices):
        raise NotImplementedError

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        c, dc_dv = self._on_vertices(self.family.vertices(p))
        return c, dc_dv @ self.family.d_vertices_dp(p)

    def value(self, p):
        return self(p)[0]


class MinEdgeLength(Constraint):
    """`|e_k| >= l_min` for every edge. A gauge fixing -- see the module docstring.

    `l_min` defaults to a tenth of the average edge, `perim/(10*n)`.
    """

    def __init__(self, family, l_min=None):
        super().__init__(family)
        self.l_min = float(l_min) if l_min is not None else family.perim/(10.0*family.n)
        if self.l_min <= 0:
            raise ValueError(f'l_min must be positive, got {self.l_min}')
        if self.l_min*family.n >= family.perim:
            raise ValueError(
                f'l_min={self.l_min:g} times n={family.n} is at least the perimeter '
                f'{family.perim:g}, so no polygon satisfies it')

    def __repr__(self):
        return f'MinEdgeLength(l_min={self.l_min:g})'

    @property
    def size(self):
        return self.family.n

    def _on_vertices(self, vertices):
        # edge k runs v_{k-1} -> v_k, matching `polygon.perimeter_gradient`'s convention
        e = vertices - np.roll(vertices, 1)
        lengths = np.abs(e)
        N = len(vertices)
        if np.any(lengths == 0):
            raise ValueError('polygon has a zero-length edge; the edge-length gradient is '
                             'undefined there. This constraint exists to keep you away from it.')
        u = e/lengths                                   # d|e_k|/dv_k, as dx + i dy

        dc = np.zeros((N, 2*N))
        k = np.arange(N)
        dc[k, k] = u.real                               # d|e_k|/dx_k
        dc[k, N + k] = u.imag                           # d|e_k|/dy_k
        dc[k, (k - 1) % N] = -u.real                    # d|e_k|/dx_{k-1}
        dc[k, N + (k - 1) % N] = -u.imag
        return lengths - self.l_min, dc


class ClosingBudget(Constraint):
    """The two conditions `polygon_vertices` raises on, as smooth inequalities.

    The free vertices fix the chain `v_1 -> ... -> v_{N-2} -> 0`; whatever perimeter is left, `C`,
    has to reach around through `v_0`. That needs `C > 0` and `C > |v_1|`. Natural in `p` rather
    than in the vertices -- `C` is a function of the free vertices only -- so this overrides
    `__call__` and needs no `d_vertices_dp`.

    `margin` keeps a strict interior; the objective's `inf` return is then a guard against a bug
    rather than a routine part of the search.
    """

    def __init__(self, family, margin=1e-6):
        super().__init__(family)
        self.margin = float(margin)

    def __repr__(self):
        return f'ClosingBudget(margin={self.margin:g})'

    @property
    def size(self):
        return 2

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        m = len(p)//2
        x, y = p[:m], p[m:]
        chain = np.concatenate([x + 1j*y, [0.0 + 0.0j]])
        seg = np.diff(chain)
        lengths = np.abs(seg)
        if np.any(lengths == 0):
            raise ValueError('two consecutive free vertices coincide; the closing budget '
                             'gradient is undefined there')
        C = self.family.perim - lengths.sum()

        # dC/dv_j: v_j appears in segments j-1 and j of the chain
        w = seg/lengths                                 # unit vectors along the chain
        dC = np.zeros(2*m)
        dC[:m] = -(np.concatenate([[0.0], w.real[:-1]]) - w.real)
        dC[m:] = -(np.concatenate([[0.0], w.imag[:-1]]) - w.imag)

        r1 = np.hypot(x[0], y[0])
        if r1 == 0:
            raise ValueError('v_1 is at the origin; the closing-budget gradient is undefined')
        dr1 = np.zeros(2*m)
        dr1[0], dr1[m] = x[0]/r1, y[0]/r1

        c = np.array([C - self.margin, C - r1 - self.margin])
        return c, np.vstack([dC, dC - dr1])


class ConstraintSet:
    """Several constraints stacked into the one `(c, J)` scipy and the trust-region step want."""

    def __init__(self, *constraints):
        self.constraints = [c for c in constraints if c is not None]

    def __repr__(self):
        return f'ConstraintSet({", ".join(repr(c) for c in self.constraints)})'

    def __len__(self):
        return sum(c.size for c in self.constraints)

    def __iter__(self):
        return iter(self.constraints)

    def __call__(self, p):
        if not self.constraints:
            p = np.asarray(p, dtype=float)
            return np.zeros(0), np.zeros((0, len(p)))
        out = [c(p) for c in self.constraints]
        return np.concatenate([o[0] for o in out]), np.vstack([o[1] for o in out])

    def value(self, p):
        return self(p)[0]

    def is_feasible(self, p, tol=0.0):
        """Never raises: an inadmissible `p` may not even describe a polygon."""
        try:
            return bool(np.all(self.value(p) >= -tol))
        except (ValueError, TypeError):
            return False

    def worst(self, p):
        """`(index, value)` of the most-violated constraint, for a message worth reading."""
        c = self.value(p)
        if c.size == 0:
            return None, np.inf
        i = int(np.argmin(c))
        return i, float(c[i])

    def as_scipy(self):
        """A `scipy.optimize.NonlinearConstraint` over the same `c >= 0`."""
        from scipy.optimize import NonlinearConstraint
        return NonlinearConstraint(lambda p: self.value(p), 0.0, np.inf,
                                   jac=lambda p: self(p)[1])
