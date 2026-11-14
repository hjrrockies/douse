"""`FourierMirrorPolygon`, and the spec's `chart` (2026-11-13): the seeded ratio runs of
thesis-lock-2 run in the Fourier chart, the mirror runs in its symmetric subspace.

The chart: every polygon it makes is exactly mirror-symmetric, its derivative is right, the
antisymmetric directions it reports really are (the two perturbed polygons are congruent), and
`from_vertices` returns a symmetric polygon unchanged and picks the same kind of axis as
`MirrorPolygon`. The plumbing: a spec's chart reaches `run_one` and `run_mirror`, a Fourier row
keeps `x` in vertex coordinates (every ledger reader assumes it) with the chart's own in
`x_chart`, and a spec without a chart hashes as before.
"""
import json
import pathlib

import numpy as np
import pytest

from douse import degeneracy as dg
from douse import spec as spec_mod
from douse.polygon import (FourierMirrorPolygon, FourierPolygon, MirrorPolygon, VertexPolygon,
                           mirror_axes, param_vector)

rm = pytest.importorskip('benchmarks.ratio_multistart')

ROOT = pathlib.Path(__file__).resolve().parent.parent
PERIM = 4.0
CASES = [(3, 'vertex'), (5, 'vertex'), (6, 'vertex'), (6, 'edge'), (7, 'vertex'), (8, 'edge')]


def _shape(v):
    return np.sort(dg.interior_angles(v)), np.sort(dg.edge_lengths(v))


@pytest.mark.parametrize('n, axis', CASES)
def test_every_polygon_is_exactly_symmetric_with_the_right_derivative(n, axis):
    fam = FourierMirrorPolygon(n, axis, perim=PERIM, s=1.0)
    q = 0.05*np.random.default_rng(n).standard_normal(n - 2)
    v = fam.vertices(q)
    defect, shift, _ = mirror_axes(v)[0]
    assert defect < 1e-14 and shift == fam.r
    assert fam.perimeter(q) == pytest.approx(PERIM, abs=1e-12)
    np.testing.assert_allclose(fam._B.T @ fam._B, np.eye(n - 2), atol=1e-14)
    h = 1e-7
    fd = np.column_stack([np.concatenate([d.real, d.imag])/(2*h) for d in
                          (fam.vertices(q + h*e) - fam.vertices(q - h*e) for e in np.eye(n - 2))])
    np.testing.assert_allclose(fam.d_vertices_dp(q), fd, atol=1e-8)


@pytest.mark.parametrize('n, axis', CASES)
def test_the_antisymmetric_directions_break_the_symmetry_and_nothing_else(n, axis):
    """Along an antisymmetric `d`, `v + t d` and `v - t d` are mirror images: congruent."""
    fam = FourierMirrorPolygon(n, axis, perim=PERIM, s=1.0)
    q = 0.05*np.random.default_rng(10 + n).standard_normal(n - 2)
    v, A = fam.vertices(q), fam.antisymmetric_basis(q)
    assert A.shape == (2*n, n - 2)
    for d in A.T:
        dz = d[:n] + 1j*d[n:]
        a, b = _shape(v + 1e-3*dz), _shape(v - 1e-3*dz)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)


def test_mirror_polygon_keeps_its_antisymmetric_basis():
    fam = MirrorPolygon(6, 'vertex', PERIM)
    q = 0.05*np.random.default_rng(3).standard_normal(4)
    v, A = fam.vertices(q), fam.antisymmetric_basis(q)
    for d in A.T:
        dz = d[:6] + 1j*d[6:]
        np.testing.assert_allclose(_shape(v + 1e-3*dz)[0], _shape(v - 1e-3*dz)[0], atol=1e-12)


@pytest.mark.parametrize('n, axis', CASES)
def test_from_vertices_returns_a_symmetric_polygon_unchanged(n, axis):
    fam = FourierMirrorPolygon(n, axis, perim=PERIM, s=1.0)
    v = fam.vertices(0.05*np.random.default_rng(20 + n).standard_normal(n - 2))
    perm = np.roll(v, 2)                                      # any labelling
    got, q, defect = FourierMirrorPolygon.from_vertices(perm, PERIM, s=1.0)
    assert defect < 1e-14 and got.axis == axis
    for x, y in zip(_shape(got.vertices(q)), _shape(v)):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_the_mirror_seeds_get_the_same_kind_of_axis_as_mirror_polygon():
    seeds = json.loads((ROOT/'benchmarks/seeds/mirror_v1.json').read_text())
    for cell, ss in seeds.items():
        for sd in ss:
            z = np.array([complex(a, b) for a, b in sd['vertices']])
            fm, _, dm = MirrorPolygon.from_vertices(z, PERIM)
            ff, qf, df = FourierMirrorPolygon.from_vertices(z, PERIM, s=1.0)
            assert ff.axis == fm.axis and df == dm, cell
            assert mirror_axes(ff.vertices(qf))[0][0] < 1e-14 and ff.x1_share(qf) > 0.5


# --- the spec's chart ---------------------------------------------------------------------------

def test_a_spec_without_a_chart_hashes_as_before():
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)])
    assert 'chart' not in sp.content() and 's' not in sp.content()
    fourier = spec_mod.RunSpec(name='t', cells=[(6, 1)], chart='fourier', s=1.0)
    assert fourier.content()['chart'] == 'fourier' and fourier.sha != sp.sha


def test_the_spec_checks_its_chart():
    with pytest.raises(ValueError, match='chart must be'):
        spec_mod.RunSpec(name='t', cells=[(6, 1)], chart='polar')
    with pytest.raises(ValueError, match='Sobolev'):
        spec_mod.RunSpec(name='t', cells=[(6, 1)], s=1.0)
    with pytest.raises(ValueError, match='convex'):
        spec_mod.RunSpec(name='t', cells=[(6, 1)], family='convex', chart='fourier', s=1.0)
    spec_mod.RunSpec(name='t', cells=[(6, 1)], family='mirror', seeds='x.json', chart='fourier',
                     s=1.0)


@pytest.mark.parametrize('name', ['mirror', 'split', 'gridseed'])
def test_the_seeded_lock2_specs_run_fourier_s1_without_bounds(name):
    sp = spec_mod.load(ROOT/f'benchmarks/specs/thesis_ratio_{name}_v1.toml')
    assert (sp.chart, sp.s, sp.angle_bounds) == ('fourier', 1.0, None)


# --- the runs -----------------------------------------------------------------------------------

def _fake(monkeypatch):
    from douse.backend import RunHealth
    from douse.optimize import OptimizeResult

    class FakeEv:
        backend, prec = 'fake', 1e-10

        def eigenvalues(self, p):
            return np.array([10.0, 20.0, 30.0]), object()

        def reset_continuation(self):
            pass

        def health(self):
            return RunHealth(worst_bound=1e-10, worst_deficit=0.05, n_replans=0, n_solves=1,
                             trace=())

    seen = {}

    def make_evaluator(N, k, fam=None, **kw):
        seen.setdefault('fams', []).append(fam)
        seen['fam'] = fam
        return FakeEv()

    def fake_maximize(ev, p0, **kw):
        seen['p0'] = np.asarray(p0)
        return OptimizeResult(x=np.asarray(p0), fun=2.5, nit=1, lams=np.array([10.0, 25.0, 30.0]),
                              gap=0.2, delta=0.01, resolution=1e-6, success=True, message='ok',
                              n_warnings=0, n_cluster_steps=0, verified=True)
    monkeypatch.setattr(rm.runs, 'make_evaluator', make_evaluator)
    monkeypatch.setattr(rm, 'maximize_ratio', fake_maximize)
    return seen


def _seed(N):
    return VertexPolygon(N, perim=PERIM).vertices(
        param_vector(FourierPolygon(N, perim=PERIM).vertices(
            0.05*np.random.default_rng(N).standard_normal(2*N - 4)), PERIM))


def test_a_fourier_spec_runs_in_the_fourier_chart_and_keeps_x_in_vertex_coordinates(monkeypatch):
    seen = _fake(monkeypatch)
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)], chart='fourier', s=1.0)
    seed = _seed(6)
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, spec=sp, seed=seed)
    assert isinstance(seen['fam'], FourierPolygon) and seen['fam'].s == 1.0
    assert row['chart'] == 'fourier' and row['s'] == 1.0
    np.testing.assert_allclose(row['x_chart'], seen['p0'])
    v = VertexPolygon(6, perim=PERIM).vertices(np.asarray(row['x']))
    for x, y in zip(_shape(v), _shape(seed)):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_a_vertex_run_row_is_unchanged(monkeypatch):
    _fake(monkeypatch)
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05)
    for key in ('chart', 's', 'x_chart', 'vertices'):
        assert key not in row


@pytest.mark.parametrize('chart, cls', [('vertex', MirrorPolygon),
                                        ('fourier', FourierMirrorPolygon)])
def test_a_mirror_spec_runs_in_its_chart(monkeypatch, chart, cls):
    seen = _fake(monkeypatch)
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)], family='mirror',
                          seeds='benchmarks/seeds/mirror_v1.json', chart=chart,
                          s=1.0 if chart == 'fourier' else 0.0)
    seed = FourierMirrorPolygon(6, 'edge', PERIM, s=1.0).vertices(np.array([.03, -.02, .01, .02]))
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, spec=sp, seed=seed, mirror=True)
    assert type(seen['fams'][0]) is cls              # the run; the probe's evaluator is vertex
    assert row['chart'] == chart and row['axis'] == 'edge'
    assert len(row['x_mirror']) == 4 and len(row['x']) == 2*6 - 4
    v = VertexPolygon(6, perim=PERIM).vertices(np.asarray(row['x']))
    for x, y in zip(_shape(v), _shape(seed)):
        np.testing.assert_allclose(x, y, atol=1e-9)
