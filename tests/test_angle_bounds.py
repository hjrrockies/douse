"""`AngleBounds`: interior-angle bounds as an optional restriction (2026-11-11).

Off by default, so the first group of tests is that nothing changes when it is off: the default
constraint set, `Convexity`'s values and gradient (its code moved into `_turning_angles`), and the
hash of every existing spec. The rest check the constraint itself and its plumbing through
`run_one`.
"""
import pathlib

import numpy as np
import pytest

from douse import FourierPolygon, VertexPolygon, runs
from douse import degeneracy as dg
from douse import spec as spec_mod
from douse.constraints import AngleBounds, ClosingBudget, Convexity, MinEdgeLength
from douse.polygon import param_vector

rm = pytest.importorskip('benchmarks.ratio_multistart')

PERIM = 4.0
SPECS = pathlib.Path(rm.__file__).resolve().parent/'specs'


def _fd(f, p, h=1e-6):
    p = np.asarray(p, dtype=float)
    cols = []
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        cols.append((f(p - 2*e) - 8*f(p - e) + 8*f(p + e) - f(p + 2*e))/(12*h))
    return np.array(cols).T


def _notched():
    """A counter-clockwise hexagon with a reflex corner (about 250 degrees) and a 32 degree one:
    start 5 of the N=6 grid, the start of go/no-go rows 14, 18 and 19."""
    return runs.start_shape(6, 5, 0, PERIM)


def _p(fam, v):
    return param_vector(v, PERIM) if isinstance(fam, VertexPolygon) else fam.from_vertices(v)


# --- nothing changes when it is off ------------------------------------------------------------

def test_default_constraint_set_is_unchanged():
    for fam, kinds in ((VertexPolygon(6, perim=PERIM), [MinEdgeLength, ClosingBudget]),
                       (FourierPolygon(6, perim=PERIM), [MinEdgeLength])):
        assert [type(c) for c in runs.constraints_for(fam)] == kinds
        assert [type(c) for c in runs.constraints_for(fam, angle_bounds=None)] == kinds
        assert [type(c) for c in runs.constraints_for(fam, angle_bounds=(None, None))] == kinds
        assert runs.constraints_for(fam).constraints[0].l_min == PERIM/(10*6)


def test_convexity_is_unchanged_by_the_refactor():
    """`Convexity` computes the turning angle exactly as it did before `_turning_angles` was
    factored out: the value is `pi - interior angle`, and the gradient matches differences."""
    fam = VertexPolygon(6, perim=PERIM)
    p = _p(fam, _notched())
    c, J = Convexity(fam, margin=0.1)(p)
    interior = dg.interior_angles(fam.vertices(p))
    assert np.allclose(c, np.pi - interior - 0.1, atol=1e-12)
    assert np.allclose(J, _fd(lambda q: Convexity(fam, margin=0.1).value(q), p), atol=1e-7)


@pytest.mark.parametrize('path', sorted(SPECS.glob('thesis_ratio_*.toml')), ids=lambda p: p.stem)
def test_existing_spec_hashes_are_unchanged(path):
    """The optional fields are dropped from `content()` when None, so no spec's identity moved.
    The pinned value is the `spec_sha` the go/no-go and rc1 rows carry, written before the change
    (`thesis_ratio_v2` with the validation's `prec_final`)."""
    import dataclasses
    sp = spec_mod.load(path)
    assert sp.angle_bounds is None and sp.edge_min is None
    for key in ('angle_min_deg', 'angle_max_deg', 'edge_min'):
        assert key not in sp.content()
    if path.stem == 'thesis_ratio_v2':
        assert dataclasses.replace(sp, prec_final=[1e-12, 1e-11, 1e-10]).sha == '2a1543964dae'


# --- the constraint ---------------------------------------------------------------------------

@pytest.mark.parametrize('chart', ['vertex', 'fourier0', 'fourier1'])
def test_values_and_gradient(chart):
    fam = (VertexPolygon(6, perim=PERIM) if chart == 'vertex'
           else FourierPolygon(6, perim=PERIM, s=float(chart[-1])))
    p = _p(fam, _notched())
    con = AngleBounds(fam, 20.0, 300.0)
    c, J = con(p)
    deg = np.degrees(dg.interior_angles(fam.vertices(p)))
    assert c.shape == (12,) and J.shape == (12, fam.n_params)
    assert np.allclose(np.degrees(c[:6]), 300.0 - deg, atol=1e-9)      # reflex side
    assert np.allclose(np.degrees(c[6:]), deg - 20.0, atol=1e-9)       # sharp side
    assert np.allclose(J, _fd(con.value, p), atol=1e-6)


def test_one_sided_and_feasibility():
    fam = VertexPolygon(6, perim=PERIM)
    p = _p(fam, _notched())
    deg = np.degrees(dg.interior_angles(fam.vertices(p)))
    assert AngleBounds(fam, min_deg=30.0).size == 6
    assert AngleBounds(fam, max_deg=260.0).size == 6
    lo, hi = deg.min(), deg.max()
    assert runs.constraints_for(fam, angle_bounds=(lo - 1, hi + 1)).is_feasible(p)
    assert not runs.constraints_for(fam, angle_bounds=(lo + 1, None)).is_feasible(p)
    assert not runs.constraints_for(fam, angle_bounds=(None, hi - 1)).is_feasible(p)


def test_max_180_is_convexity():
    fam = VertexPolygon(6, perim=PERIM)
    p = _p(fam, _notched())
    a, Ja = AngleBounds(fam, max_deg=180.0)(p)
    b, Jb = Convexity(fam)(p)
    assert np.allclose(a, b, atol=1e-14) and np.allclose(Ja, Jb, atol=1e-14)


def test_describe_and_active_constraints():
    fam = VertexPolygon(6, perim=PERIM)
    p = _p(fam, _notched())
    deg = np.degrees(dg.interior_angles(fam.vertices(p)))
    i_sharp, i_reflex = int(np.argmin(deg)), int(np.argmax(deg))
    con = AngleBounds(fam, deg[i_sharp], deg[i_reflex])          # both exactly active
    assert con.describe(i_reflex) == (i_reflex, 'max')
    assert con.describe(6 + i_sharp) == (i_sharp, 'min')
    act = runs.active_constraints(runs.constraints_for(fam, angle_bounds=(deg[i_sharp],
                                                                          deg[i_reflex])), p)
    got = {(a['vertex'], a['side']) for a in act if a['name'] == 'AngleBounds'}
    assert got == {(i_reflex, 'max'), (i_sharp, 'min')}


def test_bounds_that_admit_nothing_are_refused():
    fam = VertexPolygon(6, perim=PERIM)
    with pytest.raises(ValueError):
        AngleBounds(fam)
    with pytest.raises(ValueError):
        AngleBounds(fam, 0.0, 200.0)
    with pytest.raises(ValueError):
        AngleBounds(fam, 100.0, 90.0)
    with pytest.raises(ValueError):
        AngleBounds(fam, min_deg=121.0)             # a hexagon's angles average 120
    with pytest.raises(ValueError):
        AngleBounds(fam, max_deg=119.0)


def test_spec_fields():
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)], angle_min_deg=30.0, angle_max_deg=250.0,
                          edge_min=0.1)
    assert sp.angle_bounds == (30.0, 250.0)
    assert sp.content()['angle_min_deg'] == 30.0 and sp.content()['edge_min'] == 0.1
    assert sp.sha != spec_mod.RunSpec(name='t', cells=[(6, 1)]).sha


# --- run_one ----------------------------------------------------------------------------------

def _fake(monkeypatch, N):
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

    def fake_maximize(ev, p0, **kw):
        seen['constraints'] = kw['constraints']
        return OptimizeResult(x=np.asarray(p0), fun=2.5, nit=1, lams=np.array([10.0, 25.0, 30.0]),
                              gap=0.2, delta=0.01, resolution=1e-6, success=True, message='ok',
                              n_warnings=0, n_cluster_steps=0, verified=True)
    monkeypatch.setattr(rm.runs, 'make_evaluator', lambda *a, **k: FakeEv())
    monkeypatch.setattr(rm, 'maximize_ratio', fake_maximize)
    return seen


def test_run_one_default_row_is_unchanged(monkeypatch):
    seen = _fake(monkeypatch, 6)
    row = rm.run_one(6, 1, 5, max_iter=3, seed0=0, delta0=0.05)
    assert [type(c) for c in seen['constraints']] == [MinEdgeLength, ClosingBudget]
    for key in ('angle_bounds', 'l_min', 'active_constraints'):
        assert key not in row
    assert row['redraws'] == []


@pytest.mark.parametrize('chart', ['vertex', 'fourier'])
def test_run_one_with_bounds(monkeypatch, chart):
    seen = _fake(monkeypatch, 6)
    row = rm.run_one(6, 1, 5, max_iter=3, seed0=0, delta0=0.05, chart=chart, s=1.0,
                     angle_bounds=(30.0, 250.0))
    assert AngleBounds in [type(c) for c in seen['constraints']]
    assert row['angle_bounds'] == [30.0, 250.0] and row['l_min'] is None
    assert row['redraws'] == [] and row['active_constraints'] == []


def test_run_one_redraws_an_infeasible_start(monkeypatch):
    """Start 5 of N=6 has a 32 degree corner: a 40 degree floor redraws it, on the record."""
    _fake(monkeypatch, 6)
    row = rm.run_one(6, 1, 5, max_iter=3, seed0=0, delta0=0.05, angle_bounds=(40.0, None))
    assert row['redraws'] and row['redraws'][0]['error'].startswith('infeasible start')
    assert row['start_index'] == 5 + 1000*len(row['redraws'])
    v = runs.start_shape(6, row['start_index'], 0, PERIM)
    assert np.degrees(dg.interior_angles(v)).min() >= 40.0


def test_run_one_takes_bounds_from_the_spec(monkeypatch):
    seen = _fake(monkeypatch, 6)
    sp = spec_mod.RunSpec(name='t', cells=[(6, 1)], angle_min_deg=30.0, angle_max_deg=250.0)
    row = rm.run_one(6, 1, 5, max_iter=3, seed0=0, delta0=0.05, spec=sp)
    assert AngleBounds in [type(c) for c in seen['constraints']]
    assert row['angle_bounds'] == [30.0, 250.0]


def test_a_seeded_start_outside_the_bounds_raises(monkeypatch):
    _fake(monkeypatch, 6)
    with pytest.raises(ValueError, match='seeded start'):
        rm.run_one(6, 1, 0, max_iter=3, seed0=0, delta0=0.05, seed=_notched(),
                   angle_bounds=(40.0, None))
