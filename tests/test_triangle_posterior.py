"""`benchmarks/triangle_posterior.py` on a synthetic forward map (no grid data, no energy
integrals): the spec, the folds, the credible-region arithmetic, and a posterior that finds and
covers its truth."""
import dataclasses
import pathlib

import numpy as np
import pytest

tp = pytest.importorskip('benchmarks.triangle_posterior')
fg = pytest.importorskip('benchmarks.family_grid')
SPECS = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs'
V1 = SPECS / 'triangle_posterior_v1.toml'


def nu_fake(P):
    """Three smooth, jointly injective 'ratios' in (0, 1)."""
    a, b = P[:, 0], P[:, 1]
    return np.c_[0.3 + 0.4*a + 0.2*b, 0.2 + 0.3*b**2 + 0.1*a, 0.1 + 0.2*a*b + 0.15*b]


def E_fake(P):
    P = np.atleast_2d(P)
    return 15 + 10*((P[:, 0] - 0.5)**2 + (P[:, 1] - np.sqrt(3)/2)**2)


@pytest.fixture(scope='module')
def synth():
    g = fg.GridSpec(name='t', family='triangle', h=0.04, min_angle_deg=10.0)
    P = np.array(fg.points(g))
    sur = tp.Surrogate(P, nu_fake(P), g.h, g.min_angle_deg, refine=2, energy_fn=E_fake)
    spec = tp.PosteriorSpec(name='t', grid='benchmarks/specs/triangle_grid_v1.toml', Ks=(4,),
                            zoom=41, sigma_grid=(1e-5, 0.3, 41))
    return sur, spec


def test_v1_spec_loads_and_scheduling_is_not_content():
    spec = tp.load(V1)
    assert spec.Ks == (3, 5, 10) and spec.ig_beta == 'matched'
    assert spec.content()['grid'] == fg.load(SPECS / 'triangle_grid_v1.toml').sha
    other = dataclasses.replace(spec, slurm_time='1:00:00', tasks=3, notes='x')
    assert other.sha == spec.sha
    assert dataclasses.replace(spec, n_truths=10).sha != spec.sha


def test_unknown_keys_and_bad_sigma_priors_are_refused(tmp_path):
    p = tmp_path / 's.toml'
    p.write_text('grid = "benchmarks/specs/triangle_grid_v1.toml"\nbogus = 1\n')
    with pytest.raises(ValueError, match='unknown key'):
        tp.load(p)
    with pytest.raises(ValueError, match='ig_beta'):
        tp.PosteriorSpec(name='t', grid='g', ig_beta='vague')


def test_support_matches_the_grid_and_folds_partition_it():
    g = fg.load(SPECS / 'triangle_grid_v1.toml')
    P = np.array(fg.points(g))
    assert tp.in_support(P[:, 0], P[:, 1], g.min_angle_deg).all()
    assert not tp.in_support(np.array([0.3, 0.6, 0.0]), np.array([0.02, 0.3, 0.9]), 10).any()
    f = tp.folds(P, g.h)
    assert set(f) == {0, 1, 2, 3} and min(np.bincount(f)) > 500
    inner = tp.interior(P, g.h)
    assert 0.85 < inner.mean() < 0.95


def test_log_lik_is_the_gaussian_at_a_single_sigma():
    rng = np.random.default_rng(1)
    y, M, tau = rng.random(3), rng.random((5, 3)), np.array([1e-3, 2e-3, 0.0])
    s = np.array([0.05])
    lm, _ = tp.log_lik(y, M, tau, s, np.zeros(1))
    v = s[0]**2 + tau**2
    ref = -0.5*(((y - M)**2)/v).sum(1) - 0.5*np.log(2*np.pi*v).sum()
    assert np.allclose(lm, ref)


def test_hpd_levels():
    logd = np.log([0.2, 0.5, 0.3])
    mass = np.array([0.2, 0.5, 0.3])
    lev, t = tp.hpd(logd, mass, np.log(0.4))
    assert np.allclose(lev, [1.0, 0.5, 0.8]) and abs(t - 0.5) < 1e-15
    assert tp.hpd(logd, mass, np.log(0.6))[1] == 0.0


def test_matched_sigma_prior_peaks_near_the_true_sigma(synth):
    _, spec = synth
    spec = dataclasses.replace(spec, ig_alpha=2.0, ig_beta='matched')
    for sigma in (1e-3, 1e-2):
        s, lw = tp.sigma_weights(spec, sigma)
        assert abs(np.exp(lw).sum() - 1) < 1e-12
        assert 0.5 < s[np.argmax(lw)]/sigma < 2


def test_posterior_finds_its_truth(synth):
    sur, spec = synth
    spec = dataclasses.replace(spec, zoom=121)      # a noise-free truth: the narrowest posterior
    r = np.flatnonzero(sur.interior)[len(sur.P)//3]
    f = int(sur.fold[r])
    y = sur.nu[r]
    res = tp.posterior(sur, spec, f, y, 4, 0.0, 1e-3, sur.P[r], sur.E_grid[r], cells=True)
    c = res['cells']
    assert abs(c['mass'].sum() - 1) < 1e-12
    assert res['err_mean'] < 5e-3
    assert 0 <= res['hpd_level'] <= 1
    assert res['n90'] >= tp.MIN_N90
    assert res['kendall90'] < 0.01
    assert res['zoom_cells'] > 0


def test_credible_regions_cover_truths_drawn_from_the_prior(synth):
    """Simulation-based calibration in miniature: truths from Kendall over the interior grid
    points, data with noise of the matched size, the Kendall prior. Loose bounds, 60 truths."""
    sur, spec = synth
    spec = dataclasses.replace(spec, ig_alpha=2.0, ig_beta='matched')
    rng = np.random.default_rng(3)
    cand = np.flatnonzero(sur.interior)
    w = tp.kendall(sur.P[cand])
    sigma = 0.02
    lev = []
    for r in rng.choice(cand, 60, replace=True, p=w/w.sum()):
        s_true = sigma*np.exp(0.3*rng.standard_normal())     # within the prior's spread
        y = sur.nu[r] + s_true*rng.standard_normal(3)
        res = tp.posterior(sur, spec, int(sur.fold[r]), y, 4, 0.0, sigma, sur.P[r], sur.E_grid[r])
        lev.append(res['hpd_level'])
    lev = np.array(lev)
    assert 0.75 <= np.mean(lev <= 0.9) <= 1.0
    assert 0.3 <= np.mean(lev <= 0.5) <= 0.7


def test_items_are_a_function_of_the_spec(synth):
    sur, spec = synth
    spec = dataclasses.replace(spec, n_truths=5, truth_betas=(0.0, 1.0))
    sur.ids = np.arange(len(sur.P))
    a, b = tp.items(spec, sur), tp.items(spec, sur)
    assert a == b and len(a) == 3 + 10
    assert all(sur.interior[it[3]] for it in a)
    assert tp.noise(spec, 'map0').tolist() == tp.noise(spec, 'map0').tolist()


def test_the_dry_run_writes_one_array_job(capsys):
    assert tp.main(['slurm', str(V1), '--dry-run', '--mail', '']) == 0
    text = capsys.readouterr().out
    assert '#SBATCH --array=0-19' in text and '--account=siallocation' in text
    assert '-m benchmarks.triangle_posterior run benchmarks/specs/triangle_posterior_v1.toml' in text


def test_equal_area_chart_round_trips_and_its_height_is_normalized_area():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, .5, 3000), rng.uniform(.02, .86, 3000)
    keep = tp.in_support(a, b, 10)
    a, b = a[keep], b[keep]
    ph, q = tp.to_chart(a, b)
    a2, b2 = tp.from_chart(ph, q)
    assert np.abs(a2 - a).max() < 1e-12 and np.abs(b2 - b).max() < 1e-12
    l2 = 1 + a**2 + b**2 + (1 - a)**2 + b**2
    assert np.abs(q - 4*np.sqrt(3)*(b/2)/l2).max() < 1e-12
    assert np.allclose(tp.to_chart(0.5, 0.5)[0], 60) and np.allclose(tp.to_chart(0.2, 0.6)[0], 0)
    assert 0 < tp.support_area(10, n=200) < 60


def test_procrustes_recovers_a_similar_relabelled_mirror_image():
    t = np.array([0, 1, 0.3 + 0.5j])
    t = (t - t.mean())/np.linalg.norm(t - t.mean())
    from douse.repulsive import apex
    z = 2.5*np.exp(1j)*np.conj(t)[[2, 0, 1]] + 3     # moved, scaled, reflected, relabelled
    fit = tp.procrustes(apex(z), t)
    assert np.abs(fit - t).max() < 1e-12
