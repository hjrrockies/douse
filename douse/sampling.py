"""Random polygons for multi-start searches, sampled by the features that matter to the solver.

WHY THIS IS NOT A DETAIL. The star-shaped sampler these searches started with -- sorted random
angles about a centre, radii uniform on `[0.55, 1]` -- is simple by construction and says nothing
about the GEOMETRY it produces. Measured over the eight N=6 starts actually used: interior angles
up to 215 degrees, i.e. reentrant, on 2 of 8. That mattered, because reentrance is what defeats a
frozen basis plan: of the 79 stored multi-start rows, 7 of the 13 begun at a reentrant shape were
excluded on `worst_sigma` against 2 of the other 66. A sixth of the compute went into starts the
guards then discarded, and which starts those were was decided by an unexamined side effect of
the radius distribution.

WHAT DECIDES A CORNER'S COST. `lappy.basis_plan` sizes a corner's Fourier-Bessel block as
`M ~ (nu_osc + nu_cont)/alpha` with `alpha = pi/omega`, so the term demand scales as `omega`, and
the independence ceiling `_fb_ceiling ~ 1/alpha` moves the same way. Sharp corners are handled at
the other end, by `sharp_ref/alpha` shrinking the arc a corner owns toward `M = 1`. So the
INTERIOR ANGLE is the variable a sampler should be aware of, at both ends, and a sampler that
controls a radius instead is controlling it only through a side effect.

TWO SAMPLERS, and the honest split between them:

`random_convex` is Valtr's algorithm: uniform over convex polygons with `N` vertices, exactly and
without rejection. Use it when the question is about convex N-gons. Its angle distribution is
then a FACT about that uniform measure rather than a choice -- see `shape_stats` and the table in
`benchmarks/sampler_stats.py`.

`random_star_shaped` is the original construction, kept because the reentrant part of the family
is in scope for `lappy` and will be for the inverse problem. It now takes an explicit
`max_interior_deg`, so "how reentrant may a start be" is a parameter of the experiment instead of
an accident of the radii.

Both accept `min_interior_deg`, which rejects SPIKES -- the other end of the same axis, and the
end no convexity constraint addresses.
"""
import numpy as np

__all__ = ['random_convex', 'random_star_shaped', 'shape_stats']


def _rescale(v, perim):
    """Translate to the centroid and scale to the requested perimeter."""
    v = v - v.mean()
    return v*perim/np.abs(v - np.roll(v, 1)).sum()


def shape_stats(v):
    """`(min_interior_deg, max_interior_deg, edge_ratio)` -- the three numbers that predict cost.

    `edge_ratio` is shortest over longest edge: 1 at a regular polygon, and small where a corner
    block's owned arc shrinks with the edge that bounds it.
    """
    from . import degeneracy as dg
    a = np.degrees(dg.interior_angles(v))
    e = np.abs(np.asarray(v, dtype=complex) - np.roll(np.asarray(v, dtype=complex), 1))
    return float(a.min()), float(a.max()), float(e.min()/e.max())


def random_convex(N, perim=4.0, rng=None, min_interior_deg=0.0, min_edge=0.0,
                  max_tries=1000):
    """A uniformly random CONVEX `N`-gon of perimeter `perim`, by Valtr's algorithm.

    Sample `N` x-increments and `N` y-increments that each sum to zero (two sorted uniform chains
    differenced against each other), pair them at random, sort the resulting edge vectors by
    angle, and accumulate. Sorting by angle is what makes the result convex; the construction is
    exact, so there is no rejection step and no tuning constant -- which is the reason to prefer
    it to "sample something and throw away the non-convex ones", whose induced measure nobody
    can state.

    `min_interior_deg` rejects spikes and `min_edge` rejects slivers; between them they are the
    only place a rejection loop enters, and both default to off. `min_edge` matters more here
    than it looks: uniform-convex polygons grow slivers as `N` rises, and 17.3% of N=12 draws
    already violate `MinEdgeLength`'s default floor of `perim/(10 n)` -- a start born infeasible.
    """
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(max_tries):
        xs, ys = np.sort(rng.random(N)), np.sort(rng.random(N))
        # split each sorted chain in two and difference them in opposite directions, so the
        # increments sum to zero exactly
        def chain(s):
            lo, hi = s[0], s[-1]
            mask = rng.random(N - 2) < 0.5
            a = np.concatenate([[lo], s[1:-1][mask], [hi]])
            b = np.concatenate([[lo], s[1:-1][~mask], [hi]])
            return np.concatenate([np.diff(a), -np.diff(b)])
        dx, dy = chain(xs), chain(ys)
        rng.shuffle(dy)
        e = dx + 1j*dy
        e = e[np.argsort(np.angle(e))]
        v = np.cumsum(e)
        if len(v) != N or np.any(np.abs(v - np.roll(v, 1)) == 0):
            continue
        v = _rescale(v, perim)
        lo, _, _ = shape_stats(v)
        if lo >= min_interior_deg and np.abs(v - np.roll(v, 1)).min() >= min_edge:
            return v
    raise RuntimeError(f'no convex {N}-gon with min interior angle >= {min_interior_deg} deg and '
                       f'min edge >= {min_edge:g} in {max_tries} tries')


def random_star_shaped(N, perim=4.0, rng=None, r_lo=0.55, r_hi=1.0, gap_frac=0.4,
                       min_interior_deg=0.0, max_interior_deg=360.0, min_edge=0.0,
                       max_tries=1000):
    """Star-shaped and simple by construction, with reentrance bounded by an explicit parameter.

    NOT THE SAMPLER `benchmarks/ratio_multistart.start_shape` USES. That one is left exactly as
    it was, because a sweep resumes by start index and a ledger whose rows came from two
    different generators is worthless. This is the sampler for new runs; the recorded cells keep
    theirs.

    THE ANGULAR GAPS ARE SAMPLED DIRECTLY, NOT REJECTED INTO SHAPE. The legacy sampler draws
    sorted uniform angles and rejects any draw with a gap below 0.25 rad, which is a lottery
    whose odds collapse with `N`: measured, `N = 12` needs ~400 draws and fails outright inside
    `max_tries = 1000`, and `N = 16` is hopeless (the floor eats 64% of the circle). Since N = 12
    and 16 are the scope target, that is a trap sitting directly on the path. Here the floor is
    subtracted first and what remains of `2 pi` is split by a flat Dirichlet, so every draw is
    admissible, the floor is exact, and the distribution is uniform on the set it claims to be.
    `gap_frac` is the floor as a fraction of the mean gap `2 pi/N`, so it scales with `N` on its
    own.

    `max_interior_deg` bounds reentrance: 360 leaves it unbounded, 180 asks for convexity (at a
    rejection rate that rises steeply with `N` -- use `random_convex` for that, which needs no
    rejection at all).
    """
    rng = np.random.default_rng() if rng is None else rng
    if not 0 <= gap_frac < 1:
        raise ValueError(f'gap_frac must lie in [0, 1), got {gap_frac}')
    floor = gap_frac*2*np.pi/N
    for _ in range(max_tries):
        gaps = floor + (2*np.pi - N*floor)*rng.dirichlet(np.ones(N))
        ang = np.concatenate([[0.0], np.cumsum(gaps)[:-1]]) + rng.uniform(0, 2*np.pi)
        v = _rescale(rng.uniform(r_lo, r_hi, N)*np.exp(1j*ang), perim)
        lo, hi, _ = shape_stats(v)
        if (lo >= min_interior_deg and hi <= max_interior_deg
                and np.abs(v - np.roll(v, 1)).min() >= min_edge):
            return v
    raise RuntimeError(f'no star-shaped {N}-gon with interior angles in '
                       f'[{min_interior_deg}, {max_interior_deg}] deg in {max_tries} tries')
