"""`benchmarks/inverse_multistart.py`: the runs, the noise, the shape error. No eigensolve here."""
import json
import pathlib

import numpy as np
import pytest

im = pytest.importorskip('benchmarks.inverse_multistart')
SPECS = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs'

SPEC = """name = "t"
cases = [{name = "rect", kind = "rectangle", L = 1.2, H = 0.8, K = [5, 6]},
         {name = "tri", kind = "polygon", vertices = [[0, 0], [1, 0], [0.3, 0.5]], K = [3]}]
sigmas = [0.0, 1e-3]
noise_draws = 2
starts = 3
"""


def _spec(tmp_path, text=SPEC):
    (tmp_path / 's.toml').write_text(text)
    return im.load(tmp_path / 's.toml')


def test_runs_are_case_K_sigma_draw_start_and_sigma_zero_has_one_draw(tmp_path):
    spec = _spec(tmp_path)
    rs = im.runs(spec)
    # rect: 2 K x (1 + 2 draws) x 3 starts; tri: 1 K x 3 x 3
    assert len(rs) == 2*3*3 + 1*3*3
    assert rs[0] == (0, 5, 0.0, 0, 0) and rs[3] == (0, 5, 1e-3, 0, 0)
    assert len({r for r in rs}) == len(rs)


def test_the_hash_ignores_scheduling_and_notes_but_not_the_cases(tmp_path):
    a = _spec(tmp_path)
    b = _spec(tmp_path, SPEC + 'notes = "x"\ntasks = 7\nslurm_mem = "1G"\n')
    c = _spec(tmp_path, SPEC.replace('K = [3]', 'K = [4]'))
    assert a.sha == b.sha and a.sha != c.sha
    with pytest.raises(ValueError, match='unknown key'):
        _spec(tmp_path, SPEC + 'start = 2\n')
    with pytest.raises(ValueError, match='K >= 2'):
        _spec(tmp_path, SPEC.replace('K = [3]', 'K = [1]'))


def test_the_rectangle_spectrum_is_the_closed_form():
    lam = im.rectangle_spectrum(1.2, 0.8, 4)
    want = sorted(np.pi**2*(m**2/1.44 + k**2/0.64) for m in range(1, 5) for k in range(1, 5))[:4]
    np.testing.assert_allclose(lam, want, rtol=1e-15)


def test_case_vertices_are_ccw_at_the_spec_perimeter():
    v = im.case_vertices(dict(kind='polygon', vertices=[[0, 0], [0.3, 0.5], [1, 0]]), 4.0)
    assert im._signed_area(v) > 0 and im._perimeter(v) == pytest.approx(4.0)


def test_noise_is_relative_seeded_and_absent_at_sigma_zero():
    lam = np.array([10.0, 20.0, 30.0])
    assert np.array_equal(im.noisy(lam, 0.0, 1), lam)
    a, b = im.noisy(lam, 1e-2, 5), im.noisy(lam, 1e-2, 5)
    assert np.array_equal(a, b) and not np.array_equal(a, im.noisy(lam, 1e-2, 6))
    assert np.all(np.abs(a/lam - 1) < 0.1)


def test_congruence_error_ignores_motions_reflection_and_relabelling():
    rng = np.random.default_rng(0)
    w = np.array([0, 1.0, 1.3 + 0.7j, 0.2 + 0.9j, -0.3 + 0.4j])
    z = (np.conj(w)[::-1]*np.exp(0.7j) + (2 - 1j))
    z = np.roll(z, 2)
    err, zm = im.congruence_error(z, w)
    assert err < 1e-14
    np.testing.assert_allclose(zm, w, atol=1e-13)
    bumped = w + 1e-3*(rng.normal(size=5) + 1j*rng.normal(size=5))
    e2, _ = im.congruence_error(bumped, w)
    assert 1e-5 < e2 < 1e-3


def test_collect_summarizes_by_least_loss_start(tmp_path):
    spec = _spec(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    rows = []
    for i, (ci, K, sigma, draw, start) in enumerate(im.runs(spec)):
        if ci != 1:
            continue
        # start 1 fits best and is the right shape; start 2 is a worse local fit
        loss, err = {0: (1e-6, 0.2), 1: (1e-20, 1e-9), 2: (1e-4, 0.3)}[start]
        rows.append(dict(id=i, case='tri', N=3, K=K, sigma=sigma, draw=draw, start=start,
                         determined=True, status='ok', loss=loss, shape_err=err,
                         recovered=err < im.RECOVERED, wall=1.0, spec_sha=spec.sha))
    rows.append(dict(rows[0], id=999, spec_sha='other'))
    (out / 'task_0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    _, text = im.collect(spec, out)
    s = [x for x in im.summarize(spec, im._load(spec, out)[0]) if x['case'] == 'tri']
    assert [x['sigma'] for x in s] == [0.0, 1e-3]
    assert all(x['best_err_max'] == 1e-9 for x in s)
    assert s[0]['recovered'] == 1 and s[1]['recovered'] == 2 and s[1]['runs'] == 6
    assert '1 row(s) from another spec_sha ignored' in text


def test_the_pilot_spec_loads_and_every_case_is_determined():
    spec = im.load(SPECS / 'inverse_pilot_v1.toml')
    for case in spec.cases:
        N = len(im.case_vertices(case, spec.perim))
        assert min(case['K']) - 1 >= 2*N - 4, case['name']
