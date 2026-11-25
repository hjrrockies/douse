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


# --- random mirror seeds (2026-11-14, final_ratio_mirror_v1) ------------------------------------

ms = pytest.importorskip('benchmarks.make_seeds')


@pytest.mark.parametrize('N', [5, 6, 7, 8, 11, 12, 16])
def test_random_mirror_seeds_are_exactly_symmetric_with_the_intended_axis(N):
    seeds = ms.random_mirror_seeds(N, 3, n=6)
    assert len(seeds) == 6
    for i, (v, note) in enumerate(seeds):
        axis = 'vertex' if (N % 2 or i % 2 == 0) else 'edge'
        assert axis in note
        assert mirror_axes(v)[0][0] < 1e-14
        fam, q, defect = FourierMirrorPolygon.from_vertices(v, PERIM, s=1.0)
        assert defect < 1e-14 and fam.axis == axis
        assert np.min(dg.edge_lengths(v)) >= PERIM/(10*N)
        assert np.sum(dg.edge_lengths(v)) == pytest.approx(PERIM)
        assert VertexPolygon(N, perim=PERIM).is_valid(param_vector(v, PERIM))


def test_random_mirror_seeds_are_deterministic_and_the_same_for_every_k():
    a, b = ms.random_mirror_seeds(9, 1), ms.random_mirror_seeds(9, 7)
    for (va, _), (vb, _) in zip(a, b):
        np.testing.assert_array_equal(va, vb)
    assert not np.allclose(a[2][0], ms.random_mirror_seeds(9, 1, seed0=1)[2][0])


# --- the probe after a mirror run (2026-11-25) ---------------------------------------------------

def _mirror_spec_and_seed():
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)], family='mirror',
                          seeds='benchmarks/seeds/mirror_v1.json', chart='fourier', s=1.0)
    seed = FourierMirrorPolygon(6, 'edge', PERIM, s=1.0).vertices(np.array([.03, -.02, .01, .02]))
    return sp, seed


def test_a_mirror_row_reaches_the_ledger_before_its_probe(monkeypatch):
    """The F2 failure: a 146 h probe after the optimization, and the row only after it."""
    _fake(monkeypatch)
    sp, seed = _mirror_spec_and_seed()
    early = []
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, spec=sp, seed=seed, mirror=True,
                     on_row=early.append)
    assert len(early) == 1 and early[0]['antisym']['pending']
    assert 'pending' not in row['antisym'] and row['antisym']['n_probes'] > 0
    for key in ('fun', 'x', 'x_mirror', 'vertices', 'final_case', 'spec_sha'):
        assert early[0][key] == row[key]


def test_a_mirror_probe_stops_at_the_deadline(monkeypatch):
    """A deadline already passed (the fake optimizer ignores it): the probe does no solve."""
    import time
    _fake(monkeypatch)
    sp, seed = _mirror_spec_and_seed()
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, spec=sp, seed=seed, mirror=True,
                     deadline=time.time() - 1)
    a = row['antisym']
    assert a['out_of_time'] and a['n_done'] == 0 and a['verdict'] == 'no evaluations: out of time'
    assert a['F_full'] is None


def test_a_salvaged_row_is_the_run_s_own_row(monkeypatch):
    """`benchmarks.salvage_record` rebuilds a run's row from the record's head, last iterate and
    result -- here a record made from a real (fake-evaluator) run's row."""
    sv = pytest.importorskip('benchmarks.salvage_record')
    _fake(monkeypatch)
    sp, seed = _mirror_spec_and_seed()
    row = rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, spec=sp, seed=seed, mirror=True)
    v = np.array(row['vertices'])
    rows = [dict(kind='head', N=6, k=1, start=0, chart='fourier', s=1.0, prec=1e-9,
                 backend='fake', seeded=True, mirror=True, spec='t', spec_sha=row['spec_sha']),
            dict(kind='iter', it=1, prec=1e-10, geometry=dict(v=[v[:, 0].tolist(),
                                                                 v[:, 1].tolist()])),
            dict(kind='result', x=row['x_mirror'], fun=row['fun'], nit=row['nit'],
                 final_case=row['final_case'], convergence=row['convergence'], t=1.0)]
    got = sv.salvage_row(rows, 'task_0.jsonl.start0.record.jsonl')
    assert got['antisym']['skipped'] and got['prec'] == 1e-10
    for key in ('fun', 'nit', 'x_mirror', 'axis', 'chart', 's', 'final_case', 'convergence',
                'spec_sha', 'N', 'k', 'start', 'family'):
        assert got[key] == row[key], key
    for key in ('vertices', 'edges', 'angles'):
        np.testing.assert_allclose(got[key], row[key], atol=1e-12)
    rows[1]['geometry']['v'][0][0] += 0.01            # an iterate that is not the result's
    with pytest.raises(ValueError, match="not result.x's"):
        sv.salvage_row(rows, 'r')
    assert sv.salvage_row(rows[:2], 'r') is None      # no result: nothing to salvage
