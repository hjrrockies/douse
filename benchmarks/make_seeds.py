"""Seed files for `ratio_multistart` (`seeds` in a run spec): starts aimed at a question rather
than drawn at random.  (2026-11-03)

    uv run python -m benchmarks.make_seeds split  --cells N6_k4 N7_k5 --out benchmarks/seeds/split_v1.json
    uv run python -m benchmarks.make_seeds mirror --cells N5_k3 N8_k5 --out benchmarks/seeds/mirror_v1.json
    uv run python -m benchmarks.make_seeds grid   --cells N3_k4 N3_k5 --out benchmarks/seeds/grid_v1.json

`k` is the ledger's (0-based: N6_k4 is lam_5/lam_1). The rows the seeds come from are the best
certified rows of the ledgers under `--from` (default: thesis_ratio_v1 and v2), filtered as
`summarize_grid` filters them.

SPLIT: the max over N-gons is at least the max over (N-1)-gons, since an (N-1)-gon is an N-gon
with a flat vertex. Two cells (N6 lam_5/lam_1, N7 lam_6/lam_1) certified BELOW their (N-1)-gon
value, so no start found the (N-1)-gon's basin. These seeds are the (N-1)-gon optimum made into an
N-gon: each corner cut (a short edge across it) and each edge bent outward at its midpoint. Each
starts at about the (N-1)-gon's value and is free to improve on it.

MIRROR: the best certified rows of a cell whose optimum is nearly symmetric. `ratio_multistart`
symmetrizes each about its best mirror line and runs in `MirrorPolygon`.

GRID: the local maxima of `lam_{k+1}/lam_1` over the triangle grid (`family_grid`,
triangle_grid_v1), as triangles. The grid cross-check found every N=3 optimum within one grid
spacing of the grid's best triangle, but a neighbour-slope bound could not exclude other regions
for lam_5/lam_1 and lam_6/lam_1; a local run from each grid maximum settles them.
"""
import argparse
import json
import pathlib

import numpy as np

from douse import VertexPolygon, ledger, spectral_guard
from douse.polygon import polygon_area

from . import summarize_grid as sg

ROOT = pathlib.Path(__file__).resolve().parent
PERIM = 4.0
SIGMA_MAX = 1e-6
FROM = [ROOT/'run/slurm/thesis_ratio_v1', ROOT/'run/slurm/thesis_ratio_v2']


def parse_cell(c):
    N, k = c.split('_')
    return int(N[1:]), int(k[1:])


def certified_rows(N, k, roots=FROM):
    """The cell's certified rows over `roots`, best first: a clean bound, no missed mode, and
    the optimizer's own first-order certificate."""
    rows = []
    for root in roots:
        paths = sg.cell_ledgers(str(root)).get(f'N{N}_k{k}', [])
        rows += [ledger.normalize(r) for r in sg.load_cell(paths) if r.get('ok', True)]
    rows = [r for r in rows if r.get('worst_bound') is not None and r['worst_bound'] <= SIGMA_MAX
            and not sg.row_deficit(r) > spectral_guard.DEFICIT_TOL and r.get('success')]
    return sorted(rows, key=lambda r: -r['fun'])


def vertices(row):
    return VertexPolygon(row['N'], perim=PERIM).vertices(np.asarray(row['x']))


def _ccw(v):
    return v if polygon_area(v) > 0 else v[::-1]


def corner_cuts(u, l_min):
    """The N-gons made from the (N-1)-gon `u` by cutting one corner: vertex `i` becomes two
    points on its edges, far enough apart that the new edge clears `l_min`."""
    u, n, out = _ccw(u), len(u), []
    for i in range(n):
        a, b = u[i - 1] - u[i], u[(i + 1) % n] - u[i]
        theta = abs(np.angle(b/a))
        eps = 1.5*l_min/(2*np.sin(theta/2))
        if eps > 0.4*min(abs(a), abs(b)):
            continue
        cut = [u[i] + eps*a/abs(a), u[i] + eps*b/abs(b)]
        out.append(np.concatenate([u[:i], cut, u[i + 1:]]))
    return out


def edge_bends(u, l_min, bend=0.03):
    """The N-gons made by putting a vertex at an edge's midpoint, pushed outward by `bend` x the
    edge length (a flat vertex would be the (N-1)-gon itself, which the optimizer can only leave
    through a kink)."""
    u, n, out = _ccw(u), len(u), []
    for i in range(n):
        e = u[(i + 1) % n] - u[i]
        if abs(e)/2 < 1.2*l_min:
            continue
        mid = u[i] + e/2 - 1j*e*bend               # -i e points outward on a CCW polygon
        out.append(np.concatenate([u[:i + 1], [mid], u[i + 1:]]))
    return out


def split_seeds(N, k, roots=FROM):
    rows = certified_rows(N - 1, k, roots)
    if not rows:
        raise SystemExit(f'no certified N{N - 1}_k{k} row to split')
    best = rows[0]
    u = vertices(best)
    l_min = PERIM/(10.0*N)                       # `MinEdgeLength`'s default for N-gons
    seeds = [(v, f'N{N - 1}_k{k} {best.get("spec", "?")} start {best["start"]} '
                 f'(F={best["fun"]:.10f}): corner {j} cut') for j, v in enumerate(corner_cuts(u, l_min))]
    seeds += [(v, f'N{N - 1}_k{k} {best.get("spec", "?")} start {best["start"]} '
                  f'(F={best["fun"]:.10f}): edge {j} bent') for j, v in enumerate(edge_bends(u, l_min))]
    return seeds


def mirror_seeds(N, k, roots=FROM, n=2):
    rows = certified_rows(N, k, roots)[:n]
    return [(vertices(r), f'N{N}_k{k} {r.get("spec", "?")} start {r["start"]} (F={r["fun"]:.10f})')
            for r in rows]


def grid_seeds(N, k, grid=ROOT/'run/family_grid/triangle_grid_v1',
               spec_path=ROOT/'specs/triangle_grid_v1.toml'):
    from scipy.spatial import cKDTree
    from . import family_grid as fg
    if N != 3:
        raise SystemExit('grid seeds are triangles: N must be 3')
    spec = fg.load(spec_path)
    rows, _ = fg.collect(spec, grid)
    P, L, *_ = fg.to_arrays(spec, rows)
    R = L[:, k]/L[:, 0]
    T = cKDTree(P)
    loc = [i for i in range(len(P))
           if all(R[i] >= R[j] for j in T.query_ball_point(P[i], 1.5*spec.h) if j != i)]
    loc.sort(key=lambda i: -R[i])
    return [(fg._tri(*P[i]), f'triangle_grid_v1 local max a={P[i, 0]:.2f} b={P[i, 1]:.2f} '
                             f'(lam_{k + 1}/lam_1={R[i]:.8f})') for i in loc]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['split', 'mirror', 'grid'])
    ap.add_argument('--cells', nargs='+', required=True, help='target cells, e.g. N6_k4')
    ap.add_argument('--from', dest='roots', nargs='+', default=None,
                    help='ledger roots for split/mirror (default: thesis_ratio_v1 and v2)')
    ap.add_argument('--grid', default=str(ROOT/'run/family_grid/triangle_grid_v1'),
                    help='the triangle grid ledgers, for grid seeds')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    roots = [pathlib.Path(r) for r in args.roots] if args.roots else FROM
    make = {'split': lambda N, k: split_seeds(N, k, roots),
            'mirror': lambda N, k: mirror_seeds(N, k, roots),
            'grid': lambda N, k: grid_seeds(N, k, pathlib.Path(args.grid))}[args.kind]
    table = {}
    for c in args.cells:
        N, k = parse_cell(c)
        table[c] = [dict(vertices=[[float(z.real), float(z.imag)] for z in v], note=note)
                    for v, note in make(N, k)]
        print(f'{c}: {len(table[c])} seed(s)')
        for s in table[c]:
            print(f'    {s["note"]}')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
