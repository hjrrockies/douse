"""Is the high sigma STALENESS (plan frozen at the start) or the SHAPE itself?

For each high-sigma endpoint, plan a fresh basis AT that endpoint and measure the tension there.
Stale  => fresh plan gives ~1e-10.  Out of reach => fresh plan is still ~1e-4.
"""
import json, glob, os, warnings, numpy as np
from douse.polygon import VertexPolygon
from douse.evaluate import SpectrumEvaluator
from douse import degeneracy as dg

PERIM = 4.0
rows = []
for f in sorted(glob.glob('benchmarks/run/ratio_multistart/N*_k*.jsonl')):
    for line in open(f):
        d = json.loads(line); d['_cell'] = os.path.basename(f)[:-6]; rows.append(d)

bad = sorted([d for d in rows if (d.get('worst_sigma') or 0) > 1e-6], key=lambda d: -d['worst_sigma'])
ok  = [d for d in rows if 0 < (d.get('worst_sigma') or 0) < 1e-9 and d.get('success')]
ok  = sorted(ok, key=lambda d: -d['N'])[:2]

print(f"{'cell':7} {'st':>2} {'maxang':>7} {'run_sigma':>10} {'fresh_sigma':>12} {'F_run':>13} {'F_fresh':>13}")
for d in bad[:5] + ok:
    x = np.array(d['x'], float)
    fam = VertexPolygon(d['N'], perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=d['k'] + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*d['k'] + 4), replan_tol=None, plan_at=x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams, _ = ev.eigenvalues(x)
        sig = ev.sigma_at(x, lams[0])
    a = np.degrees(dg.interior_angles(fam.vertices(x)))
    F = lams[d['k']]/lams[0]
    print(f"{d['_cell']:7} {d['start']:>2} {a.max():7.1f} {d['worst_sigma']:10.1e} {sig:12.1e} "
          f"{d['fun']:13.9f} {F:13.9f}")
