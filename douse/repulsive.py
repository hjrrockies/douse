"""The repulsive shape prior: a tangent-point energy of polygons.  (2026-11-03)

    prior(dP)  propto  exp(-beta * E(P)) * Kendall(dP),   P simple.

Vendored from the notebook `jupyter-notebooks/repulsive_prior/tangent_point.py` (the arclength-
weighted `energy` only) so the thesis runs do not depend on notebook code. `energy` integrates
one kernel over all pairs of boundary points (x, y):

    k(x, y) = |T_x x (y - x)|^a / |y - x|^b * (D(x, y) / |y - x|)^c,

T_x the unit tangent at x, D the chord (P/pi) sin(pi g/P) of the arclength g between x and y.
With (a, b, c) = (2, 3.5, 1): ordinary corners are finite (b < a + 2), folds and contacts are
infinite (b + c > a + 1, c >= a + 2 - b), a straight angle costs nothing, and the factor
P^(b - a - 2) makes E scale invariant. The regular square has E = 9.957, the equilateral
triangle 15.04.

`kendall_apex_density` is Kendall's uniform shape measure for triangles in apex coordinates
(vertices 0, 1, a + ib): proportional to (centroid size)^-4, the Bookstein-chart density.
"""
import warnings
from functools import lru_cache

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import roots_jacobi

__all__ = ['energy', 'apex', 'kendall_apex_density', 'A_EXP', 'B_EXP', 'C_EXP']

A_EXP = 2.0      # numerator exponent a
B_EXP = 3.5      # denominator exponent b, in (a+1, a+2)
C_EXP = 1.0      # arclength weight c >= a + 2 - b (contacts infinite)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1)
_GL_W = 0.5 * _GL_W

_NEAR_X, _NEAR_W = np.polynomial.legendre.leggauss(16)
_NEAR_X = 0.5 * (_NEAR_X + 1)
_NEAR_W = 0.5 * _NEAR_W
_NEAR_CUTS = (1.5, 4.0, 9.0)   # panels in the sinh variable: the peak, then its tails
_NEAR_GROW = 2.0  # panel growth away from the outer integral's breakpoints
_NEAR = 0.5       # non-adjacent edges closer than this times the shorter edge use _near_pair_w


def _chord(gap, P):
    """Intrinsic distance of two boundary points `gap` apart in arclength: the chord of a circle
    of circumference P. Same zeros as min(gap, P - gap), equal to it to O(gap^3), smooth."""
    return (P / np.pi) * np.sin(np.pi * gap / P)


@lru_cache(maxsize=8)
def _jacobi(d2, n=16):
    """Nodes/weights for int_0^1 v^(d2-1) F(v) dv."""
    x, w = roots_jacobi(n, 0.0, d2 - 1.0)
    return (1 + x) / 2, w / 2 ** d2


def _adj_w(l1, l2, cth, sth, P, a, b, c):
    """int_0^l1 ds int_0^l2 dt (t sth)^a / r^b * (D/r)^c,  r = |t u2 - s u1|,  D = chord(s + t).

    In polar coordinates (s, t) = rho (cos psi, sin psi) the kernel is rho^(a-b) h(psi) g(rho k),
    g(u) = (chord(u)/u)^c smooth: the radial integral is Gauss-Jacobi with the rho^(a-b+1)
    weight built in, and one angular integral is left for quad.
    """
    if sth <= 0.0:
        return np.inf if cth > 0 else 0.0
    d2 = a - b + 2.0
    v, wv = _jacobi(d2)
    psi_c = np.arctan2(l2, l1)

    def f(psi):
        cs, sn = np.cos(psi), np.sin(psi)
        q = 1.0 - 2.0 * cs * sn * cth
        R = l1 / cs if psi < psi_c else l2 / sn
        kap = cs + sn
        u = R * kap * v
        g = (_chord(u, P) / u) ** c
        return (sn * sth) ** a / q ** (b / 2) * (kap / np.sqrt(q)) ** c * R ** d2 * (wv @ g)

    tot = 0.0
    for lo, hi in ((0.0, psi_c), (psi_c, np.pi / 2)):
        pts = [np.pi / 4] if lo < np.pi / 4 < hi else None
        with warnings.catch_warnings():   # near-folds: quad reports roundoff at ~1e-9 relative
            warnings.simplefilter('ignore', IntegrationWarning)
            tot += quad(f, lo, hi, points=pts, limit=200, epsabs=0, epsrel=1e-10)[0]
    return tot


def _closest(p, e, q, f):
    """Closest parameters (s, t) in [0,1]^2 and distance of segments p + s e, q + t f; d = 0 if
    they cross."""
    cr = lambda u, v: (u.conjugate() * v).imag
    w = q - p                     # only differences of vertices: rounding follows the polygon's
    den = cr(e, f)                # size, not its distance from the origin
    if den != 0:
        s = cr(w, f) / den
        t = cr(w, e) / den
        if 0 <= s <= 1 and 0 <= t <= 1:
            return s, t, 0.0

    def proj(x, g):
        return float(np.clip((x * g.conjugate()).real / abs(g) ** 2, 0, 1))
    cands = []
    for s in (0.0, 1.0):
        t = proj(s * e - w, f)
        cands.append((abs(s * e - w - t * f), s, t))
    for t in (0.0, 1.0):
        s = proj(w + t * f, e)
        cands.append((abs(s * e - w - t * f), s, t))
    d, s, t = min(cands)
    return s, t, d


def _graded(L, s0, h, hmin=1e-7):
    """Gauss-Legendre nodes/weights on [0, 1], panels growing geometrically away from s0 from
    width h (absolute length; the edge has length L)."""
    h = max(h / L, hmin)
    br = {0.0, 1.0, s0}
    k = h
    while s0 - k > 0 or s0 + k < 1:
        if s0 - k > 0:
            br.add(s0 - k)
        if s0 + k < 1:
            br.add(s0 + k)
        k *= 2
    br = np.array(sorted(br))
    lo, hi = br[:-1], br[1:]
    x = (lo[:, None] + (hi - lo)[:, None] * _GL_X[None, :]).ravel()
    w = ((hi - lo)[:, None] * _GL_W[None, :]).ravel()
    return x, w


def _edge_frame(z, e, l, i, j, t):
    """Points of edge j at arclength t, in the frame of edge i (origin at its start, its tangent
    along the real axis): the coordinate along edge i and the distance from its line."""
    T = e[i].conjugate() / l[i]
    w, step = (z[j] - z[i]) * T, e[j] / l[j] * T
    return w.real + t * step.real, np.abs(w.imag + t * step.imag)


def _pair_w(z, e, l, cum, P, i, j, s0, t0, d, a, b, c):
    """Ordered pair (x on edge i, y on edge j), non-adjacent, with the arclength weight."""
    s, ws = _graded(l[i], s0, d)
    t, wt = _graded(l[j], t0, d)
    f, h = _edge_frame(z, e, l, i, j, t * l[j])
    ar = np.hypot(f[None, :] - (s * l[i])[:, None], h[None, :])
    gap = np.abs((cum[i] + s * l[i])[:, None] - (cum[j] + t * l[j])[None, :])
    D = _chord(gap, P)
    k = h[None, :] ** a / ar ** b * (D / ar) ** c
    return l[i] * l[j] * np.einsum('i,ij,j', ws, k, wt)


def _near_pair_w(z, e, l, cum, P, i, j, d, a, b, c):
    """Ordered pair (x on edge i, y on edge j) of edges a distance d apart, d small against both.

    `_pair_w` grades toward one closest point, but nearly parallel edges are close along their
    whole overlap: its nodes then face each other across the gap and it sees d^(a-b-c) instead of
    the true d^(a-b-c+1). Here x is integrated separately for each y, with nodes clustered at the
    foot of y (a sinh map of width dist(y, edge i)), so the nearly singular part is resolved at
    every y. What is left as a function of y only changes quickly within d of the ends of edge j
    and of the projections of edge i's ends, where the y panels are graded.
    """
    U = e[j] / l[j]
    foot = [((p - z[j]) * U.conjugate()).real for p in (z[i], z[(i + 1) % len(z)])]
    br = {0.0, l[j]}
    for t0 in [0.0, l[j]] + [float(np.clip(f, 0, l[j])) for f in foot]:
        br.add(t0)
        k = d / 2
        while k < l[j]:
            br.update(t for t in (t0 - k, t0 + k) if 0 < t < l[j])
            k *= _NEAR_GROW
    br = np.array(sorted(br))
    lo, hi = br[:-1], br[1:]
    t = (lo[:, None] + (hi - lo)[:, None] * _NEAR_X[None, :]).ravel()
    wt = ((hi - lo)[:, None] * _NEAR_W[None, :]).ravel()

    f, h = _edge_frame(z, e, l, i, j, t)
    f0 = np.clip(f, 0, l[i])
    dist = np.hypot(f - f0, h)
    tot = np.zeros(len(t))
    for side, room in ((-1.0, f0), (1.0, l[i] - f0)):
        W = np.arcsinh(room / dist)
        cuts = [np.zeros_like(W)] + [np.minimum(c, W) for c in _NEAR_CUTS] + [W]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            w = lo[:, None] + (hi - lo)[:, None] * _NEAR_X[None, :]
            off = side * dist[:, None] * np.sinh(w)      # x - foot, never formed by subtraction
            ws = dist[:, None] * np.cosh(w) * (hi - lo)[:, None] * _NEAR_W[None, :]
            r = np.hypot((f - f0)[:, None] - off, h[:, None])
            D = _chord(np.abs(cum[i] + f0[:, None] + off - (cum[j] + t)[:, None]), P)
            tot += np.sum(ws * h[:, None] ** a / r ** b * (D / r) ** c, axis=1)
    return wt @ tot


def energy(z, a=A_EXP, b=B_EXP, c=C_EXP, parts=False):
    """Scale-invariant, refinement-invariant tangent-point energy of the polygon with vertices z
    (complex, either orientation). inf if the polygon is not simple.

    parts=True returns (E, corner part, rest): the corner part is every pair of points on two
    edges that share a vertex, the rest every other pair."""
    z = np.asarray(z, complex)
    n = len(z)
    e = np.roll(z, -1) - z
    l = np.abs(e)
    P = l.sum()
    cum = np.r_[0.0, np.cumsum(l)[:-1]]
    E_c = 0.0
    for i in range(n):              # vertex i+1, shared by edges i and i+1
        j = (i + 1) % n
        u1, u2 = -e[i] / l[i], e[j] / l[j]
        cth = (u1.conjugate() * u2).real
        sth = abs((u1.conjugate() * u2).imag)
        for l1, l2 in ((l[i], l[j]), (l[j], l[i])):
            E_c += _adj_w(l1, l2, cth, sth, P, a, b, c)
        if not np.isfinite(E_c):
            return (np.inf, np.inf, np.inf) if parts else np.inf
    E_r = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            s0, t0, d = _closest(z[i], e[i], z[j], e[j])
            if d == 0.0:
                return (np.inf, E_c, np.inf) if parts else np.inf
            if d < _NEAR * min(l[i], l[j]):
                E_r += _near_pair_w(z, e, l, cum, P, i, j, d, a, b, c)
                E_r += _near_pair_w(z, e, l, cum, P, j, i, d, a, b, c)
            else:
                E_r += _pair_w(z, e, l, cum, P, i, j, s0, t0, d, a, b, c)
                E_r += _pair_w(z, e, l, cum, P, j, i, t0, s0, d, a, b, c)
    scale = P ** (b - a - 2)
    E = (E_c + E_r) * scale
    return (E, E_c * scale, E_r * scale) if parts else E


def apex(z):
    """Triangle -> apex a+ib with the longest side as base [0,1], 0 <= a <= 1/2, b > 0."""
    l = np.abs(np.roll(z, -1) - z)
    i = int(np.argmax(l))
    p, q, r = z[i], z[(i + 1) % 3], z[(i + 2) % 3]
    w = (r - p) / (q - p)
    if w.imag < 0:
        w = w.conjugate()
    if w.real > 0.5:
        w = 1 - w.conjugate()
    return w


def kendall_apex_density(w):
    """Kendall's uniform shape measure in apex coordinates (unnormalized)."""
    return (1 + np.abs(2 * w - 1)**2 / 3) ** -2
