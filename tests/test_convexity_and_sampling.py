"""The convex search mode: the constraint, the sampler, and the separation between the two modes.

Reentrant N-gons stay in scope -- this is the machinery for asking the CONVEX question on its
own, which the unconstrained sweep cannot answer because a start that begins reentrant is what
defeats the frozen basis plan in the first place.
"""
import numpy as np
import pytest

from douse import Convexity, VertexPolygon, param_vector
from douse import degeneracy as dg
from douse.sampling import random_convex, random_star_shaped, shape_stats

PERIM = 4.0


def regular(N):
    return PERIM/N/(2*np.sin(np.pi/N))*np.exp(2j*np.pi*np.arange(N)/N)


@pytest.mark.parametrize('N', [4, 5, 6, 7])
def test_the_jacobian_matches_finite_differences(N):
    fam = VertexPolygon(N, perim=PERIM)
    rng = np.random.default_rng(N)
    p = param_vector(regular(N)*(1 + 0.15*rng.normal(size=N)), PERIM)
    con = Convexity(fam)
    c, J = con(p)
    Jn = np.zeros_like(J)
    h = 1e-6
    for j in range(len(p)):
        e = np.zeros_like(p)
        e[j] = h
        Jn[:, j] = (con.value(p + e) - con.value(p - e))/(2*h)
    assert np.abs(J - Jn).max() < 1e-7


@pytest.mark.parametrize('N', [4, 5, 6, 7])
def test_the_constraint_IS_the_turning_angle(N):
    """Not the cross product: scale-free, so `margin` means the same thing at every vertex."""
    fam = VertexPolygon(N, perim=PERIM)
    p = param_vector(regular(N)*(1 + 0.1*np.random.default_rng(N).normal(size=N)), PERIM)
    assert np.allclose(Convexity(fam).value(p), dg.turning_angles(fam.vertices(p)))


def test_a_reentrant_shape_is_infeasible_and_a_convex_one_is_not():
    fam = VertexPolygon(5, perim=PERIM)
    v = np.array([0, 2, 1 + 1j, 2 + 2j, 0 + 2j], dtype=complex)      # one deep notch
    assert np.degrees(dg.interior_angles(v)).max() > 180
    assert Convexity(fam).value(param_vector(v, PERIM)).min() < 0
    assert Convexity(fam).value(param_vector(regular(5), PERIM)).min() > 0


def test_a_flat_vertex_is_admissible_at_margin_zero_which_keeps_every_M_gon_reachable():
    """The reachability argument: a convex M-gon lives in the N-gon family with flat vertices."""
    fam = VertexPolygon(5, perim=PERIM)
    v = np.array([0, 1, 2, 2 + 2j, 0 + 2j], dtype=complex)           # vertex 1 is flat
    turn = dg.turning_angles(v)
    assert np.isclose(np.abs(turn).min(), 0.0, atol=1e-12)
    assert Convexity(fam).value(param_vector(v, PERIM)).min() >= -1e-12
    assert Convexity(fam, margin=0.01).value(param_vector(v, PERIM)).min() < 0


def test_a_margin_that_no_polygon_can_satisfy_is_refused_up_front():
    fam = VertexPolygon(5, perim=PERIM)
    with pytest.raises(ValueError, match='2 pi'):
        Convexity(fam, margin=2*np.pi/5)


@pytest.mark.parametrize('N', [4, 5, 6, 8, 12])
def test_valtr_draws_are_convex_simple_and_closed(N):
    rng = np.random.default_rng(N)
    for _ in range(50):
        v = random_convex(N, PERIM, rng)
        assert len(v) == N
        turn = dg.turning_angles(v)
        assert turn.min() > -1e-12
        assert np.isclose(turn.sum(), 2*np.pi)
        assert np.isclose(np.abs(v - np.roll(v, 1)).sum(), PERIM)
        assert VertexPolygon(N, perim=PERIM).is_valid(param_vector(v, PERIM))


@pytest.mark.parametrize('N', [4, 6, 12])
def test_the_feasibility_bands_are_actually_enforced(N):
    """A start born below `MinEdgeLength`'s floor is infeasible before the run begins."""
    rng = np.random.default_rng(N)
    floor = PERIM/(10.0*N)
    for _ in range(30):
        v = random_convex(N, PERIM, rng, min_interior_deg=15.0, min_edge=floor)
        lo, hi, _ = shape_stats(v)
        assert lo >= 15.0 and hi <= 180.0
        assert np.abs(v - np.roll(v, 1)).min() >= floor


@pytest.mark.parametrize('N', [12, 16])
def test_the_star_shaped_sampler_SURVIVES_THE_SCOPE_TARGET(N):
    """The legacy rejection loop cannot: its 0.25 rad floor eats 48% of the circle at N=12 and
    64% at N=16, so the draw essentially never lands. The gap floor is subtracted first here."""
    rng = np.random.default_rng(N)
    for _ in range(20):
        v = random_star_shaped(N, PERIM, rng)
        assert len(v) == N
        assert np.isclose(dg.turning_angles(v).sum(), 2*np.pi)


def test_the_two_modes_write_to_different_ledgers():
    """Pooling a constrained optimum into an unconstrained answer is the failure this prevents."""
    rm = pytest.importorskip('benchmarks.ratio_multistart')
    import argparse
    import os
    ns = argparse.Namespace(N=6, k=2)
    plain = os.path.join(rm.OUT, f'N{ns.N}_k{ns.k}.jsonl')
    convex = os.path.join(rm.OUT, f'N{ns.N}_k{ns.k}_convex.jsonl')
    assert plain != convex
    assert rm.convex_start_shape(6, 0, 0).shape == rm.start_shape(6, 0, 0).shape
    v = rm.convex_start_shape(6, 3, 0)
    assert np.degrees(dg.interior_angles(v)).max() <= 180.0
