"""Which levels' derivatives are determined only together: the eps/gap test.  (2026-11-08)

Davis-Kahan, in Moler-Payne's form: a set of levels' joint eigenspace is good to about eps/gap,
with `eps` its worst residual and `gap` its distance to the rest. chladni's clusters catch
eigenfunctions that visibly overlap; this also catches a pair rotated coherently within its span,
which is orthogonal and wrong (the regular hexagon with a vertex moved 1e-6: Hadamard good to 7e-6
at prec 1e-10, with no cluster).
"""
import numpy as np
import pytest
from chladni.eigensolver import Cluster, Level, Spectrum

from douse.chladni_backend import ChladniSolver, SpectrumEvaluator
from douse.cluster_opt import _active_cluster
from douse.derivative import partition_levels
from douse.polygon import VertexPolygon


def spec(lams, eps, cluster_of=(), bound=1e-14):
    clu = {i: 0 for i in cluster_of}
    levels = tuple(Level(l, 1, bound, mp_eps=e, cluster=clu.get(i))
                   for i, (l, e) in enumerate(zip(lams, eps)))
    clusters = ((Cluster(levels=tuple(cluster_of), mult=len(cluster_of), spread=1e-10,
                         overlap=1e-6),) if cluster_of else ())
    return Spectrum(levels, clusters=clusters)


def test_a_pair_closer_than_its_residual_allows_is_grouped_and_a_separated_one_is_not():
    # the hexagon's numbers: lam 18.13, relative gap 4e-7, eps 1e-11 -> x = 2.5e-5
    s = spec([7.15, 18.1316680, 18.1316755, 32.45], [1e-11]*4)
    assert partition_levels(s, 1e-6) == [(0, 1), (1, 3), (3, 4)]


def test_the_target_decides_how_much_is_grouped():
    s = spec([10.0, 10.001, 12.0], [1e-11]*3)
    assert partition_levels(s, 1e-6) == [(0, 1), (1, 2), (2, 3)]        # x = 1e-7
    assert partition_levels(s, 1e-8) == [(0, 2), (2, 3)]


def test_grouping_reaches_a_fixed_point():
    # (0,1) merge on level 0's eps; the merged group's eps then reaches level 2, which its own
    # residual alone would not have
    s = spec([10.0, 10.001, 10.011], [1e-8, 1e-12, 1e-12])
    assert partition_levels(s, 1e-6) == [(0, 3)]


def test_a_level_with_no_residual_gives_no_evidence():
    s = spec([10.0, 10.000001, 12.0], [np.nan, np.nan, 1e-11])
    assert partition_levels(s, 1e-6) == [(0, 1), (1, 2), (2, 3)]


def test_chladnis_clusters_seed_the_partition_however_loose_the_target():
    s = spec([10.0, 20.0, 20.0 + 2e-9, 30.0], [1e-14]*4, cluster_of=(1, 2))
    assert partition_levels(s, 1e-1) == [(0, 1), (1, 3), (3, 4)]


def test_a_gap_inside_the_error_bars_is_no_gap():
    s = spec([10.0, 10.0 + 1e-9, 12.0], [1e-13]*3, bound=1e-8)
    assert partition_levels(s, 1e-1) == [(0, 2), (2, 3)]


def test_the_active_set_takes_whole_groups():
    s = spec([7.0, 18.1316680, 18.1316755, 32.45], [1e-11]*4)
    assert _active_cluster(list(s.unfolded), 1, 1e-12, spec=s) == (1, 2), 'no target: as before'
    assert _active_cluster(list(s.unfolded), 1, 1e-12, spec=s, tau_g=1e-6) == (1, 3)
    assert _active_cluster(list(s.unfolded), 2, 1e-12, spec=s, tau_g=1e-6) == (1, 3)
    assert _active_cluster(list(s.unfolded), 3, 1e-12, spec=s, tau_g=1e-6) == (3, 4)


def test_the_active_set_never_takes_lam_1_for_a_group():
    # the ratio's denominator is simple: a group reaching it must not make it a cluster member
    s = spec([10.0, 10.0 + 1e-5, 12.0], [1e-11]*3)
    assert partition_levels(s, 1e-6) == [(0, 2), (2, 3)]
    assert _active_cluster(list(s.unfolded), 1, 1e-12, spec=s, tau_g=1e-6) == (1, 2)


class _Rec:
    def __init__(self, s):
        self.spectrum = s

    def hadamard(self, i, V, report=False):
        H = np.full((4, 1, 1), 10.0*(i + 1))
        return (H, 'r') if report else H


def _solved(s):
    sol = ChladniSolver(_Rec(s), np.array([0, 1, 1j, -1]), 0.0)
    sol._velocity = lambda: None
    return sol


def test_jacobian_rows_of_a_group_are_its_mean():
    s = spec([10.0, 10.00001, 12.0], [1e-11]*3)
    J = _solved(s).dlams_dvertices(list(s.unfolded), 2, degenerate='mean', groups=[(0, 2)])
    assert np.allclose(J[:2], 15.0) and np.allclose(J[2], 30.0)


def test_without_groups_the_rows_stay_the_levels_own():
    s = spec([10.0, 10.00001, 12.0], [1e-11]*3)
    J = _solved(s).dlams_dvertices(list(s.unfolded), 2, degenerate='mean')
    assert np.allclose(J[0], 10.0) and np.allclose(J[1], 20.0)


@pytest.mark.parametrize('target,sigma,want', [
    ([10.0, 10.0 + 1e-8, 12.0], 0.0, [(0, 2)]),      # the target does not resolve the pair
    ([10.0, 11.0, 12.0], 0.0, []),                   # it does: rows must keep the difference
    ([10.0, 10.0 + 5e-4, 12.0], 1e-3, [(0, 2)]),     # resolved only inside the noise
    ([10.0, 11.0, 12.0], 1e-3, []),
])
def test_only_a_group_the_target_cannot_resolve_gets_mean_rows(target, sigma, want):
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, degenerate='mean', degenerate_tol=0.0)
    ev.set_target(target, sigma=sigma, tau_g=1e-6)
    s = spec([10.0, 10.000001, 12.0], [1e-11]*3)
    assert ev.mean_groups(s) == want


def test_no_target_no_groups():
    ev = SpectrumEvaluator(VertexPolygon(4), n_eigs=3, degenerate='mean', degenerate_tol=0.0)
    s = spec([10.0, 10.000001, 12.0], [1e-11]*3)
    assert ev.mean_groups(s) == []
