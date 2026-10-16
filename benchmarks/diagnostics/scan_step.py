"""How much gain is left near an endpoint? The objective scanned along the model step, fit by a parabola.

At a ledger endpoint, build the local model with the active cluster tied to a radius of 1e-4
(`eps_abs = 1e-4 * scale`), take its unit step direction `u`, solve at `p + t u` for `t` from
1e-5 to 5e-4, and least-squares fit `dF(t) = s t + c t^2`. With `c < 0` the nearby maximum along
`u` is at `t* = -s/(2c)` and the gain left is `-s^2/(4c)`: the certificate to quote at a kink,
where criticality (`gain/delta` of the first-order model) does not go to zero.

The 2026-10-15 STALL ANALYSIS (finding 1, seed 2 windowed is a genuine local maximum: slope
2.4e-4, curvature -2.0, 7e-9 of gain left) came from

    uv run python -m benchmarks.diagnostics.scan_step 2 windowed2

Its printed output was not kept; the numbers are quoted from the session's plan file.
"""
import numpy as np

from douse.cluster_opt import local_model, trust_region_step

from benchmarks.diagnostics.endpoint import endpoint_from_args, parser, ratio

TS = (1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4)


def scan(fam, ev, cons, p, k, radius=1e-4, ts=TS, rng=None):
    """The model slope along its step direction, the samples `(t, dF)` and the parabola fit."""
    rng = np.random.default_rng(0) if rng is None else rng
    lams, solver = ev.eigenvalues(p)
    F0 = ratio(lams, k)
    m0 = local_model(fam, p, solver, lams, k=k)
    m = local_model(fam, p, solver, lams, k=k, eps_abs=radius*m0['scale'])
    d, gain = trust_region_step(m, radius, rng=rng, lin=(cons(p) if cons is not None else None))
    u = d/np.linalg.norm(d)
    samples = []
    for t in ts:
        lq, _ = ev.eigenvalues(p + t*u)
        samples.append((float(t), ratio(lq, k) - F0))
    T = np.array([s[0] for s in samples])
    D = np.array([s[1] for s in samples])
    A = np.vstack([T, T**2]).T
    (s, c), *_ = np.linalg.lstsq(A, D, rcond=None)
    out = dict(F0=F0, model_slope=float(gain/radius), samples=samples, slope=float(s),
               curvature=float(c), rms=float(np.sqrt(np.mean((A @ [s, c] - D)**2))),
               t_star=None, gain_left=None)
    if c < 0:
        out.update(t_star=float(-s/(2*c)), gain_left=float(-s*s/(4*c)))
    return out


def main():
    args = parser(__doc__).parse_args()
    e = endpoint_from_args(args)
    r = scan(e.fam, e.ev, e.cons, e.p, e.k)
    print(f"F0={r['F0']:.10f} model slope {r['model_slope']:.3e}")
    for t, dF in r['samples']:
        print(f"  t={t:.1e}  dF={dF:+.3e}  linear={r['model_slope']*t:+.3e}")
    tail = ('' if r['t_star'] is None else
            f" -> max at t*={r['t_star']:.2e}, gain left {r['gain_left']:.2e}")
    print(f"fit dF = {r['slope']:.3e} t + {r['curvature']:.3e} t^2{tail}; residual rms {r['rms']:.1e}")


if __name__ == '__main__':
    main()
