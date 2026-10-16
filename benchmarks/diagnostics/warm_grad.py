"""Do warm solves give the same gradients as cold ones? Both warm paths against a cold solve.

At a ledger endpoint `p`, solve cold at `p`, then at `p + step` (a 2e-3 move of the first
coordinate) three ways: warm on the windowed path (the evaluator continuing from `p`), warm on
the prefix path (`endpoint.PrefixEvaluator`), and cold (`reset_continuation()` first). Report
the eigenvalue differences and the max and relative differences of the gradients of `lam_1`
(`g1`) and of the cluster mean (`gmu`) from `cluster_opt.local_model`. If they agree to
round-off, a secant Hessian built from warm-solve gradients is fed accurate pairs, and a
blow-up in it is pair geometry, not gradient noise.

The 2026-10-15 STALL ANALYSIS (warm gradients match cold ones to 1e-12 relative, both paths)
came from

    uv run python -m benchmarks.diagnostics.warm_grad 0 prefix

Its printed output was not kept; the number is quoted from the session's plan file.
"""
import numpy as np

from douse.cluster_opt import local_model

from benchmarks.diagnostics.endpoint import PrefixEvaluator, endpoint_from_args, parser


def _grads(fam, ev, q, k):
    lams, solver = ev.eigenvalues(q)
    m = local_model(fam, q, solver, lams, k=k)
    return np.asarray(lams), np.asarray(m['g1'], float), np.asarray(m['gmu'], float)


def compare(fam, make_evaluator, p, k, step=None):
    """`make_evaluator()` returns a fresh evaluator. Returns the differences, warm vs cold, at
    `p + step` on both warm paths."""
    step = np.eye(len(p))[0]*2e-3 if step is None else np.asarray(step, float)
    q = p + step
    ev = make_evaluator()
    _grads(fam, ev, p, k)                                   # cold at p
    lw, g1w, gmw = _grads(fam, ev, q, k)                    # warm (windowed) at q
    ev.reset_continuation()
    lc, g1c, gmc = _grads(fam, ev, q, k)                    # cold at q
    evp = PrefixEvaluator(make_evaluator())
    _grads(fam, evp, p, k)
    lp, g1p, gmp = _grads(fam, evp, q, k)                   # warm (prefix) at q

    def diff(a, b):
        d = float(np.abs(a - b).max())
        s = float(np.abs(b).max())
        return dict(abs=d, rel=d/s if s > 0 else d)
    return dict(lam_windowed=float(np.abs(lw - lc).max()), lam_prefix=float(np.abs(lp - lc).max()),
                g1_windowed=diff(g1w, g1c), g1_prefix=diff(g1p, g1c),
                gmu_windowed=diff(gmw, gmc), gmu_prefix=diff(gmp, gmc))


def main():
    from douse import runs
    args = parser(__doc__).parse_args()
    e = endpoint_from_args(args)
    N = int(e.row['N'])
    r = compare(e.fam, lambda: runs.make_evaluator(N, e.k, fam=e.fam), e.p, e.k)
    print(f"lam diffs warm-cold at p+step: windowed {r['lam_windowed']:.2e}  prefix {r['lam_prefix']:.2e}")
    for key in ('g1_windowed', 'g1_prefix', 'gmu_windowed', 'gmu_prefix'):
        print(f"{key:14s} |diff| {r[key]['abs']:.2e}  rel {r[key]['rel']:.2e}")


if __name__ == '__main__':
    main()
