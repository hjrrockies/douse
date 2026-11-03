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
    den = cr(e, f)
    if den != 0:
        s = cr(q - p, f) / den
        t = cr(q - p, e) / den
        if 0 <= s <= 1 and 0 <= t <= 1:
            return s, t, 0.0

    def proj(x, a, g):
        return float(np.clip(((x - a) * g.conjugate()).real / abs(g) ** 2, 0, 1))
    cands = []
    for s in (0.0, 1.0):
        t = proj(p + s * e, q, f)
        cands.append((abs(p + s * e - q - t * f), s, t))
    for t in (0.0, 1.0):
        s = proj(q + t * f, p, e)
        cands.append((abs(p + s * e - q - t * f), s, t))
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


def _pair_w(z, e, l, cum, P, i, j, s0, t0, d, a, b, c):
    """Ordered pair (x on edge i, y on edge j), non-adjacent, with the arclength weight."""
    s, ws = _graded(l[i], s0, d)
    t, wt = _graded(l[j], t0, d)
    x = z[i] + s * e[i]
    y = z[j] + t * e[j]
    r = y[None, :] - x[:, None]
    ar = np.abs(r)
    gap = np.abs((cum[i] + s * l[i])[:, None] - (cum[j] + t * l[j])[None, :])
    D = _chord(gap, P)
    T = e[i] / l[i]
    k = np.abs((T.conjugate() * r).imag) ** a / ar ** b * (D / ar) ** c
    return l[i] * l[j] * np.einsum('i,ij,j', ws, k, wt)


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
