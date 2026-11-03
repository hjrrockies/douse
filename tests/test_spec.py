"""`douse.spec`: a run specification is a file, content-hashed, cited by every row."""
import pathlib

import pytest

from douse import spec as S

SPECS = pathlib.Path(__file__).resolve().parent.parent / 'benchmarks' / 'specs'


def test_the_thesis_spec_loads_and_hashes_stably():
    a = S.load(SPECS / 'thesis_ratio_v0.toml')
    b = S.load(SPECS / 'thesis_ratio_v0.toml')
    assert a.name == 'thesis_ratio_v0' and a.sha == b.sha and len(a.sha) == 12
    assert (6, 2) in a.cells and all(N >= 3 and k >= 1 for N, k in a.cells)
    assert a.backend == 'chladni' and a.prec == 1e-10 and not a.convex
    assert a.ledger_name(6, 2) == 'N6_k2.jsonl'


def test_editing_a_spec_changes_its_hash(tmp_path):
    src = (SPECS / 'thesis_ratio_v0.toml').read_text()
    (tmp_path / 'a.toml').write_text(src)
    (tmp_path / 'b.toml').write_text(src.replace('max_iter = 400', 'max_iter = 401'))
    assert S.load(tmp_path / 'a.toml').sha != S.load(tmp_path / 'b.toml').sha
    assert S.load(tmp_path / 'a.toml').sha == S.load(SPECS / 'thesis_ratio_v0.toml').sha


def test_an_unknown_key_is_an_error_not_a_default(tmp_path):
    (tmp_path / 'bad.toml').write_text('name = "x"\ncells = [[4, 1]]\nmax_iters = 3\n')
    with pytest.raises(ValueError, match='unknown key'):
        S.load(tmp_path / 'bad.toml')
    (tmp_path / 'fam.toml').write_text('name = "x"\ncells = [[4, 1]]\nfamily = "concave"\n')
    with pytest.raises(ValueError, match='family'):
        S.load(tmp_path / 'fam.toml')


def test_provenance_names_the_spec_and_both_shas():
    spec = S.load(SPECS / 'thesis_ratio_v0.toml')

    class Ev:
        backend, prec = 'chladni', 1e-10
    row = S.provenance(Ev(), spec)
    assert row['spec'] == 'thesis_ratio_v0' and row['spec_sha'] == spec.sha
    assert set(row) >= {'backend', 'prec', 'douse_sha', 'chladni_sha'}
    assert 'spec' not in S.provenance(Ev())


def test_seeds_are_hashed_by_content_and_a_spec_without_them_is_unchanged(tmp_path):
    seeds = tmp_path / 'seeds.json'
    seeds.write_text('{"N3_k1": [{"vertices": [[0, 0], [1, 0], [0.5, 0.8]], "note": "x"}]}')
    base = 'name = "x"\ncells = [[3, 1]]\n'
    (tmp_path / 'a.toml').write_text(base)
    (tmp_path / 'b.toml').write_text(base + f'seeds = "{seeds}"\n')
    a, b = S.load(tmp_path / 'a.toml'), S.load(tmp_path / 'b.toml')
    assert 'seeds' not in a.content() and a.seed_polygons(3, 1) == []
    assert b.seed_polygons(3, 1)[0][2] == 0.5 + 0.8j and b.seed_polygons(4, 1) == []
    sha = b.sha
    seeds.write_text(seeds.read_text().replace('0.8', '0.9'))
    assert S.load(tmp_path / 'b.toml').sha != sha


def test_the_mirror_family_needs_seeds_and_has_its_own_ledger(tmp_path):
    (tmp_path / 'm.toml').write_text('name = "x"\ncells = [[3, 1]]\nfamily = "mirror"\n')
    with pytest.raises(ValueError, match='seeds'):
        S.load(tmp_path / 'm.toml')
    m = S.load(SPECS / 'thesis_ratio_mirror_v1.toml')
    assert m.ledger_name(5, 3) == 'N5_k3_mirror.jsonl'
    assert all(len(m.seed_polygons(N, k)) == m.starts for N, k in m.cells)


def test_run_dir_follows_douse_runs(monkeypatch, tmp_path):
    from douse.spec import run_dir
    monkeypatch.delenv('DOUSE_RUNS', raising=False)
    assert run_dir('family_grid', 'x') == 'benchmarks/run/family_grid/x'
    monkeypatch.setenv('DOUSE_RUNS', str(tmp_path))
    assert run_dir('family_grid', 'x') == str(tmp_path/'family_grid'/'x')
    from benchmarks import family_grid as fg
    spec = fg.GridSpec(name='t', family='triangle', h=0.1)
    assert fg.default_out(spec) == str(tmp_path/'family_grid'/'t')
