"""One place that knows how a run is set up: the evaluator, the start shapes, the constraints.

Until today `regular` was defined nine times, `_perturbed` four, `start_shape` three (with three
signatures), and the evaluator was constructed in ~40 places, a dozen of them still passing
lappy-only options the chladni evaluator silently ignores (2026-10-14).
"""
import numpy as np

from .constraints import ClosingBudget, ConstraintSet, Convexity, MinEdgeLength
from .polygon import VertexPolygon
from .sampling import random_convex

__all__ = ['PERIM', 'regular', 'perturbed', 'start_shape', 'convex_start_shape',
           'constraints_for', 'make_evaluator', 'family_for']

PERIM = 4.0


def regular(N, perim=PERIM):
    """The regular N-gon of perimeter `perim`, vertices counter-clockwise from the real axis."""
    return perim/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


def perturbed(N, jitter, seed=0, perim=PERIM):
    """The regular N-gon with every vertex moved by `jitter` x its mean radius, Gaussian."""
    rng = np.random.default_rng(seed)
    v = regular(N, perim)
    return v + jitter*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()


def start_shape(N, i, seed0=0, perim=PERIM):
    """Start `i` of the multistart grid: 0 is a lightly jittered regular N-gon, the rest are
    star-shaped and random (simple by construction, which `VertexPolygon.is_valid` would
    otherwise reject at a rate that wastes the start budget at larger `N`)."""
    rng = np.random.default_rng(seed0 + 1000*i + N)
    if i == 0:
        v = regular(N, perim)
        return v + 0.02*(rng.normal(size=N) + 1j*rng.normal(size=N))*np.abs(v).mean()
    ang = np.sort(rng.uniform(0, 2*np.pi, N))
    while np.min(np.diff(np.concatenate([ang, [ang[0] + 2*np.pi]]))) < 0.25:
        ang = np.sort(rng.uniform(0, 2*np.pi, N))
    rad = rng.uniform(0.55, 1.0, N)
    v = rad*np.exp(1j*ang)
    return v*perim/np.abs(np.diff(np.concatenate([v, v[:1]]))).sum()


def convex_start_shape(N, i, seed0=0, min_interior_deg=15.0, perim=PERIM):
    """Start `i` for a CONVEX sweep: 0 is the jittered regular N-gon, the rest Valtr draws
    (uniform over convex N-gons), banded by the edge floor `MinEdgeLength` uses and by a
    minimum interior angle that keeps the spike end away from the basis plan's thin-wedge
    regime."""
    if i == 0:
        return start_shape(N, 0, seed0, perim)
    return random_convex(N, perim, np.random.default_rng(seed0 + 1000*i + N),
                         min_interior_deg=min_interior_deg, min_edge=perim/(10.0*N))


def family_for(N, perim=PERIM):
    return VertexPolygon(N, perim=perim)


def constraints_for(fam, convex=False):
    """The grid's constraints: the edge floor (a gauge fixing) and the closing budget, plus
    convexity for a convex sweep (a restriction, which is the point of that sweep)."""
    cons = [MinEdgeLength(fam), ClosingBudget(fam)]
    if convex:
        cons.append(Convexity(fam))
    return ConstraintSet(*cons)


def make_evaluator(N, k, *, n_eigs=None, prec=1e-10, backend='chladni', perim=PERIM,
                   degenerate_tol=0.0, warn_gap=0.0, plan_at=None, fam=None):
    """The evaluator a ratio run uses: `n_eigs = k + 2` by default (the k-th ratio and one
    level of headroom for the cluster), the degeneracy guard off (`maximize_ratio` handles
    degeneracies itself) and the gap warning off (these runs converge onto a crossing)."""
    fam = family_for(N, perim) if fam is None else fam
    n_eigs = (k + 2) if n_eigs is None else int(n_eigs)
    if backend == 'chladni':
        from .chladni_backend import SpectrumEvaluator
        return SpectrumEvaluator(fam, n_eigs=n_eigs, prec=prec, degenerate_tol=degenerate_tol,
                                 warn_gap=warn_gap, plan_at=plan_at)
    if backend == 'lappy':
        from .evaluate import LappySpectrumEvaluator
        return LappySpectrumEvaluator(fam, n_eigs=n_eigs, degenerate_tol=degenerate_tol,
                                      warn_gap=warn_gap, plan_at=plan_at,
                                      lam_max_window=max(8, 2*k + 4), replan_tol=1e-9)
    raise ValueError(f"backend must be 'chladni' or 'lappy', got {backend!r}")
