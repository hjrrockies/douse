"""Is the stall NOISE-limited? Measure the criticality plateau at a collapsed endpoint.

Section 9's method: criticality does not go to zero at these points, it PLATEAUS at the accuracy
floor of the shape derivatives. Re-solving the first-order model across six decades of the active
-set width shows the floor directly. If the plateau at a collapsed endpoint sits far above the
model gain at the radius where the run died, every step was being judged on noise and the
collapse is explained without appealing to the model at all.
"""
import json, warnings, numpy as np
from douse import VertexPolygon, SpectrumEvaluator
from douse.cluster_opt import local_model, model_gain
from douse import degeneracy as dg
warnings.simplefilter('ignore')

CASES = [
    ('STALLED  (re-run of N6_k1 start 2)',
     json.loads(open('benchmarks/run/ratio_multistart/rerun/N6_k1_start2.jsonl').read().strip())),
]
for l in open('benchmarks/run/ratio_multistart/N6_k1.jsonl'):
    r = json.loads(l)
    if r.get('ok', True) and r.get('success') and abs(r['fun'] - 2.5340067775) < 1e-6:
        CASES.append(('CERTIFIED (N6_k1 start 7)', r)); break

for label, row in CASES:
    fam = VertexPolygon(row['N'], perim=4.0)
    p = np.array(row['x'], float)
    ev = SpectrumEvaluator(fam, n_eigs=row['k'] + 2, warn_gap=0.0, degenerate_tol=0.0,
                           lam_max_window=max(8, 2*row['k'] + 4), replan_tol=1e-9, plan_at=p)
    lams, solver = ev.eigenvalues(p)
    a = np.degrees(dg.interior_angles(fam.vertices(p)))
    print(f"\n{label}   F={row['fun']:.9f}  max_angle={a.max():.1f}  "
          f"min_edge={min(row['edges']):.4f}")
    print(f"  {'Delta':>10} {'m':>3} {'criticality = gain/Delta':>26}")
    for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        try:
            model = local_model(fam, p, solver, lams, k=row['k'],
                                eps_abs=d*float(np.max(np.abs(lams))))
            g, _ = model_gain(model, d)
            print(f"  {d:10.0e} {model['m']:>3} {g/d:26.3e}")
        except Exception as exc:                                      # noqa: BLE001
            print(f"  {d:10.0e}   -  {type(exc).__name__}: {str(exc)[:40]}")
