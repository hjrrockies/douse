"""Why did the trust radius collapse? A from-start run logging the model and curvature state per iterate.

Run `maximize_ratio` from a seed's start shape (`runs.perturbed(N, 0.08, seed)`, as the
overnight driver does) and write one JSON line per iterate with what the optimizer computed:
the mode, the eps-active cluster and its split, the trust radius `delta`, the first-order gain
and criticality (`gain1`, `crit1`, after `_first_order_step`), the second-order gain (`gain2`,
after `_second_order_step`), both curvature objects' state (`H1` for `lam_1`, `Hmu` for the
cluster mean: ready, updates, skipped, the norm of `B`, its extreme eigenvalues, and `gamma` for
the scalar secant), then `rho`, the new `F` and `delta` after `_trial_step`. The log comes from
wrapping those two stages of `cluster_opt._STAGES` for the duration of the run (`instrumented`,
which restores them); nothing in `cluster_opt` is edited and nothing is re-solved.

`--show LOG` prints a log one line per iterate. The two logs of 2026-10-15 are

    benchmarks/run/diagnostics/sr1_blowup_seed0_prefix_2026_10_15.jsonl     (SR1: the collapse)
    benchmarks/run/diagnostics/scalar_secant_seed0_prefix_2026_10_15.jsonl  (scalar: converged at 335)

made (before `quasi_newton='scalar'` existed, the second by a monkeypatch now retired) as

    uv run python -m benchmarks.diagnostics.instrumented_run 0 prefix 400 OUT.jsonl --qn sr1
    uv run python -m benchmarks.diagnostics.instrumented_run 0 prefix 400 OUT.jsonl --qn scalar
"""
import argparse
import contextlib
import json
import time

import numpy as np

import douse.cluster_opt as CO
from douse import runs
from douse.polygon import param_vector

from benchmarks.diagnostics.endpoint import warm_path
from benchmarks.diagnostics.replay import QN


def _curvature(H):
    if H is None:
        return None
    ready = bool(H.ready)
    rec = dict(ready=ready, updates=int(H.n_updates), skipped=int(H.n_skipped),
               normB=float(np.linalg.norm(H.B)) if ready else 0.0,
               eigB=([float(x) for x in np.linalg.eigvalsh(H.B)[[0, -1]]] if ready else None))
    if hasattr(H, 'gamma'):
        rec['gamma'] = float(H.gamma)
    return rec


@contextlib.contextmanager
def instrumented(sink):
    """Within the block, every iterate of `maximize_ratio` appends a record to `sink` (a list,
    or anything with `.append`)."""
    rec = {}
    t0 = time.perf_counter()
    stock = CO._STAGES

    def second(st, o, e, tally):
        rec.clear()
        rec.update(t=round(time.perf_counter() - t0, 1), it=int(st.it), F=float(st.F),
                   mode=st.model['mode'], cluster=[int(i) for i in st.model['cluster']],
                   split=float(st.model.get('split', float('nan'))), delta=float(st.delta),
                   gain1=float(st.gain), crit1=float(st.crit), d1=float(np.linalg.norm(st.d)),
                   H1=_curvature(st.H1), Hmu=_curvature(st.Hmu))
        out = CO._second_order_step(st, o, e, tally)
        rec.update(gain2=float(st.gain), d2=float(np.linalg.norm(st.d)),
                   delta_after2=float(st.delta), outcome2=out.name)
        if out is not CO.Outcome.CONTINUE:
            sink.append(dict(rec))
        return out

    def trial(st, o, e, tally):
        out = CO._trial_step(st, o, e, tally)
        rec.update(rho=(float(st.rho) if np.isfinite(st.rho) else None), F_after=float(st.F),
                   delta_after=float(st.delta), rejects=len(st.reject_run), outcome=out.name)
        sink.append(dict(rec))
        return out

    CO._STAGES = tuple(second if s is CO._second_order_step else trial if s is CO._trial_step
                       else s for s in stock)
    try:
        yield sink
    finally:
        CO._STAGES = stock


class _JsonlSink:
    def __init__(self, fh):
        self.fh = fh

    def append(self, rec):
        self.fh.write(json.dumps(rec) + '\n')
        self.fh.flush()


def run(ev, p0, k, cons=None, max_iter=400, quasi_newton='scalar', sink=None, **kw):
    """`(result, sink)` for an instrumented run; `sink` defaults to a new list."""
    sink = [] if sink is None else sink
    with instrumented(sink):
        res = CO.maximize_ratio(ev, p0, k=k, max_iter=max_iter, constraints=cons,
                                quasi_newton=quasi_newton, **kw)
    return res, sink


def show_line(r):
    def curv(h):
        return '-' if h is None else f"{h['updates']}/{h['skipped']} {h['normB']:.1e}"
    rho = '  -inf' if r.get('rho') is None else f"{r['rho']:+.2f}"
    return (f"{r['t']:6.0f}s it={r['it']:3d} {r['mode'][:2]} cl={r['cluster']} split={r['split']:.1e} "
            f"F={r['F']:.8f} d={r['delta']:.1e} g1={r['gain1']:.2e} g2={r['gain2']:.2e} rho={rho} "
            f"H1[{curv(r['H1'])}] Hmu[{curv(r['Hmu'])}] {r['outcome2'][:4]}/{r.get('outcome', '')[:4]}")


def show(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                print(show_line(json.loads(line)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--show', metavar='LOG', help='print a log and exit')
    ap.add_argument('seed', type=int, nargs='?')
    ap.add_argument('path', choices=('windowed', 'prefix'), nargs='?')
    ap.add_argument('max_iter', type=int, nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--qn', choices=tuple(QN), default='scalar')
    ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--k', type=int, default=2)
    args = ap.parse_args()
    if args.show:
        return show(args.show)
    if args.out is None:
        ap.error('seed path max_iter out are required unless --show')
    fam = runs.family_for(args.N)
    ev = runs.make_evaluator(args.N, args.k, fam=fam)
    p0 = param_vector(runs.perturbed(args.N, 0.08, args.seed), runs.PERIM)
    with open(args.out, 'w') as fh:
        res, _ = run(warm_path(ev, args.path), p0, args.k, runs.constraints_for(fam),
                     args.max_iter, QN[args.qn], sink=_JsonlSink(fh))
    print('END', f"F={res['fun']:.10f}", 'nit', res['nit'], res['message'], flush=True)


if __name__ == '__main__':
    main()
