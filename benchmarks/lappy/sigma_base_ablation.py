"""What sets `sigma_base`, the tension plateau away from any mode? -- an ablation.

    uv run python -m benchmarks.sigma_base_ablation [--shape bad|good|both] [--out FILE.json]

WHY THIS QUANTITY. Measured on the two shapes in `jupyter_notebooks/missing_mode.ipynb`, the MPS
tension near a mode is a CLIPPED V,

    sigma(lam) = min(sigma_base, C |lam - lam_k|)

with straight flanks (widths at 30/50/70/90% of the plateau come out in the ratio 0.3:0.5:0.7:0.9)
and a slope that is universal once written in wavenumber `k = sqrt(lam)`: C_k = 2 sqrt(lam) C_lam
= 0.23 +- 0.01 over 14 measurements, two shapes, and a factor of four in `lam`. So the width of
the well, in `k`, is

    w_k = 2 q sigma_base / C_k  ~  8.3 q sigma_base

and every measured width follows from `sigma_base` alone. The floor does not matter, the slope
does not matter, `lam` does not matter. `sigma_base` = 6.4e-03 at the shape that solved and
2.5e-03 at the shape that lost three modes, and that single factor of 2.6 is the entire difference
between finding 6 modes of 6 and finding 1.

WHAT IT MEANS. `sigma_base` is the tension the basis achieves where NO solution exists -- how
badly it fails to satisfy the boundary condition when it should be unable to satisfy it at all. A
low value means the basis can nearly meet the boundary condition at a non-eigenvalue, which is
ill-conditioning rather than a property of the spectrum. So the question this script asks is:

    IS `sigma_base` A KNOB OR A PROPERTY?

Four candidate knobs, chosen because each is a documented suspect:

  rtol_call    the pencil's rank truncation, applied per call. `lappy.mps.rtol_default`'s own note
               says retaining directions below the level at which the matrix entries are known
               "injects noise into the tension curve, which is what produces spurious minima".
  rtol_built   the same constant, but set at construction so the PLANNER's conditioning ceilings
               move with it -- `default_basis_for` warns that letting the two drift apart
               "would silently mis-size those ceilings".
  rng          the interior-point draw. `from_domain` says outright that it "genuinely moves the
               answer", and sigma is a boundary-over-interior ratio, so this is the denominator.
  basis size   `lam_max` (via `lam_max_window`) and `target`, which set how many Fourier-Bessel
               terms each corner gets.

REPORTING BOTH NUMBERS IS THE POINT. A knob that raises `sigma_base` and the slope together buys
nothing, so every row carries `sigma_base`, `C_k` and the width `w_k` it implies, plus that width
against the scan's own step in `k`. `w_k/dk >= 1` would mean the search cannot miss the mode.
"""
import argparse
import json
import os
import warnings

import numpy as np

from lappy import basis_plan as BP
from lappy import mps
from lappy.asymp import weyl_est
from lappy.mps import MPSEigensolver

from douse import VertexPolygon

PERIM, N, K = 4.0, 3, 2
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'run', 'missing_mode', 'cache.npz')
FAM = VertexPolygon(N, perim=PERIM)

MASK_HALF = 2.0          # lam within this of a true mode is not "off resonance"
FINE_HALF, FINE_N = 2.5, 251


def load_shapes():
    d = np.load(CACHE, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    return {tag: dict(p=np.asarray(d[f'p_{tag}'], float),
                      fem=np.asarray(meta[tag]['fem'], float),
                      window=meta['windows'][tag]) for tag in ('good', 'bad')}


def build(p, lam_max_window=8, target=1e-14, prec=1e-13, rtol=None, rng=None):
    """`(dom, solver, plan)` -- the evaluator's own configuration, with the knobs exposed.

    Mirrors `SpectrumEvaluator.plan`/`.solver` (douse/evaluate.py:218,235) rather than calling
    them, because those deliberately freeze the plan and give no way to vary `rtol` or `rng`.
    `rtol` is threaded into BOTH the planner's config and the solver, which is the pairing
    `default_basis_for` insists on.
    """
    dom = FAM(np.asarray(p, float))
    lam_max = weyl_est(lam_max_window, dom)
    kw = {} if rtol is None else dict(cfg=BP.PlanConfig(rtol=rtol))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plan = BP.plan_basis(dom, lam_max, target=target, **kw)
        solver = MPSEigensolver.from_domain(
            dom, lam_max=lam_max, basis=BP.realize(plan, dom), prec=prec,
            rng=rng, **({} if rtol is None else dict(rtol=rtol)))
    return dom, solver, plan


def sigma_on(solver, lam, rtol=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = solver.tensions(np.asarray(lam, float), rtol=rtol,
                               n_workers=mps.n_workers_default)
    return np.array([r[0] for r in rows], float)


def measure(solver, fem, window, ref_lam, rtol=None):
    """`sigma_base`, and the dip geometry at `ref_lam`.

    `sigma_base` is measured OFF RESONANCE by construction -- a median over a grid with every
    known mode masked out -- rather than as a median of a sweep that oversamples near modes,
    which is the proxy the notebook used and which biases low.
    """
    a, b = window['a'], window['b']
    grid = np.linspace(a, b, 220)
    grid = grid[np.min(np.abs(grid[:, None] - fem[None, :]), axis=1) > MASK_HALF]
    s_off = sigma_on(solver, grid, rtol)
    base = float(np.median(s_off))

    # THE FINE WINDOW HAS TO FOLLOW THE WELL. At a high `sigma_base` the V does not reach the
    # plateau within a fixed +-2.5, every sample counts as "in the dip", and the width saturates
    # at the window itself -- which the first version of this reported as a suspiciously
    # repeated 0.2057 across unrelated configurations.
    half, s, fine = FINE_HALF, None, None
    for _ in range(4):
        fine = np.linspace(ref_lam - half, ref_lam + half, FINE_N)
        s = sigma_on(solver, fine, rtol)
        i = int(np.argmin(s))
        lam0, floor = float(fine[i]), float(s[i])
        sel = (s > 10*floor) & (s < 0.5*base) & (np.abs(fine - lam0) > 0)
        C_lam = float(np.median(s[sel]/np.abs(fine[sel] - lam0))) if sel.sum() > 5 else np.nan
        need = 3*0.9*base/C_lam if C_lam == C_lam and C_lam > 0 else half
        if not np.isfinite(need) or need <= 1.2*half:
            break
        half = min(float(need), 0.45*float(np.min(np.abs(np.delete(
            fem, np.argmin(np.abs(fem - ref_lam))) - ref_lam))))
    C_k = 2*np.sqrt(lam0)*C_lam
    below = s <= 0.9*base
    w_lam = float(fine[below].max() - fine[below].min()) if below.any() else 0.0
    saturated = bool(below.any() and (below[0] or below[-1]))
    w_k = w_lam/(2*np.sqrt(lam0))
    dk = (b - a)/window['n_pts']/(2*np.sqrt(lam0))       # the scan's step, in wavenumber
    # The ACCEPTANCE width is unsaturable: sigma crosses ttol on the V's flank, so it depends on
    # the slope alone and can be read off C without any window at all.
    w_k_ttol = 2*mps.ttol_default/C_lam/(2*np.sqrt(lam0))
    return dict(sigma_base=base, floor=floor, C_k=C_k, w_k=w_k, w_over_dk=w_k/dk,
                w_k_ttol=w_k_ttol, saturated=saturated, half=float(half),
                lam0=lam0, n_off=int(len(grid)))


HEAD = (f'{"config":>24} {"n_bas":>6} {"sigma_base":>11} {"floor":>9} {"C_k":>6} '
        f'{"w_k(vis)":>9} {"w_k/dk":>7}  findable?  resolves?')


def row(name, plan, m):
    """One line, with the two questions that matter spelled out rather than left to the reader.

    `findable` -- the visible catchment is at least one scan step wide, so the grid cannot step
    over the mode. `resolves` -- the floor still clears `ttol`, so a bracket that IS found is
    accepted. A configuration needs BOTH, and the default has only the second.
    """
    findable = m['w_over_dk'] >= 1.0
    resolves = m['floor'] <= mps.ttol_default
    return (f'{name:>24} {plan.n_total:>6} {m["sigma_base"]:>11.3e} {m["floor"]:>9.1e} '
            f'{m["C_k"]:>6.3f} {m["w_k"]:>9.4f}{"*" if m["saturated"] else " "}'
            f'{m["w_over_dk"]:>7.3f}  {"YES" if findable else " no":>8}  '
            f'{"YES" if resolves else " no":>8}')


def run_shape(tag, sh, out):
    p, fem, window = sh['p'], sh['fem'], sh['window']
    inwin = fem[(fem >= window['a']) & (fem <= window['b'])]
    ref = float(inwin[1]) if len(inwin) > 1 else float(inwin[0])
    print(f'\n{"="*88}\n{tag.upper()} shape  p={np.round(p, 6).tolist()}  '
          f'reference mode lam={ref:.2f}  window [{window["a"]:.1f}, {window["b"]:.1f}]'
          f'\n{"="*88}')

    dom, solver, plan = build(p)
    print(plan.summary())
    print(f'  capped={plan.capped}  shortfall={plan.shortfall!r}  '
          f'n_fb={plan.n_fb} n_fs={plan.n_fs} n_total={plan.n_total}')
    print(f'\n{HEAD}')
    res = []

    # (A) the pencil truncation, per call -- same solver, same basis, no rebuild
    for rt in (None, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6):
        m = measure(solver, fem, window, ref, rtol=rt)
        nm = f'rtol_call={"default" if rt is None else f"{rt:.0e}"}'
        print(row(nm, plan, m)); res.append(dict(knob='rtol_call', value=rt, **m))

    # (B) the same constant, but set at construction so the PLANNER's ceilings move with it
    for rt in (1e-11, 1e-9, 1e-7):
        _, s2, pl2 = build(p, rtol=rt)
        m = measure(s2, fem, window, ref)
        print(row(f'rtol_built={rt:.0e}', pl2, m))
        res.append(dict(knob='rtol_built', value=rt, n_basis=pl2.n_total, **m))

    # (C) the interior-point draw -- the denominator of the ratio sigma measures
    for seed in (0, 1, 2, 3):
        _, s2, pl2 = build(p, rng=seed)
        m = measure(s2, fem, window, ref)
        print(row(f'int_pts rng={seed}', pl2, m))
        res.append(dict(knob='rng', value=seed, **m))

    # (D) basis size, through the two dials that set it
    for w in (6, 8, 12, 16):
        _, s2, pl2 = build(p, lam_max_window=w)
        m = measure(s2, fem, window, ref)
        print(row(f'lam_max_window={w}', pl2, m))
        res.append(dict(knob='lam_max_window', value=w, n_basis=pl2.n_total, **m))
    for tg in (1e-8, 1e-10, 1e-12, 1e-14):
        _, s2, pl2 = build(p, target=tg)
        m = measure(s2, fem, window, ref)
        print(row(f'target={tg:.0e}', pl2, m))
        res.append(dict(knob='target', value=tg, n_basis=pl2.n_total, **m))

    out[tag] = dict(p=p.tolist(), ref_lam=ref, rows=res,
                    plan=dict(n_fb=plan.n_fb, n_fs=plan.n_fs, n_total=plan.n_total,
                              capped=bool(plan.capped), shortfall=str(plan.shortfall),
                              corners=[dict(alpha=float(c.alpha), M=int(c.M),
                                            nu_osc=float(c.nu_osc), nu_cont=float(c.nu_cont))
                                       for c in plan.corners]))


VERIFY = [('default', {}),
          ('rtol_built=1e-11', dict(rtol=1e-11)),
          ('rtol_built=1e-10', dict(rtol=1e-10)),
          ('lam_max_window=12', dict(lam_max_window=12)),
          ('lam_max_window=10', dict(lam_max_window=10)),
          ('target=1e-12', dict(target=1e-12)),
          ('int_pts rng=3', dict(rng=3))]


def verify(tag, sh):
    """Does a WIDER well actually recover the modes? The ablation measures geometry; this runs
    the search and checks the answer against FEM.

    A wide well is necessary and not sufficient -- widening it by wrecking the basis would move
    every located value off its true position, which is why the accuracy column is here.
    """
    p, fem, window = sh['p'], sh['fem'], sh['window']
    a, b, n_pts = window['a'], window['b'], int(window['n_pts'])
    truth = fem[(fem >= a) & (fem <= b)]
    print(f'\n{"="*88}\nVERIFY on the {tag.upper()} shape: run the real scan over '
          f'[{a:.1f}, {b:.1f}] at {n_pts} points\n'
          f'  {len(truth)} true modes in the window: {np.round(truth, 2).tolist()}\n{"="*88}')
    print(f'{"config":>20} {"n_bas":>6} {"found":>7} {"max rel err":>12} {"n_rej":>6}  missed')
    for name, kw in VERIFY:
        try:
            _, solver, plan = build(p, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                eigs, mults, _, rej = solver.solve_interval(a, b, n_pts, return_rejected=True)
            got = np.repeat(np.asarray(eigs, float), np.asarray(mults, int))
        except Exception as exc:                                   # a config may simply fail
            print(f'{name:>20} {"":>6} {type(exc).__name__}: {str(exc)[:40]}')
            continue
        hit, err, missed = 0, 0.0, []
        for t in truth:
            if len(got) and np.min(np.abs(got - t)) <= 0.03*t:
                hit += 1
                err = max(err, float(np.min(np.abs(got - t))/t))
            else:
                missed.append(round(float(t), 1))
        print(f'{name:>20} {plan.n_total:>6} {f"{hit}/{len(truth)}":>7} {err:>12.2e} '
              f'{len(rej):>6}  {missed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true',
                    help='run the real scan under each promising configuration')
    ap.add_argument('--shape', default='both', choices=('bad', 'good', 'both'))
    ap.add_argument('--out', default=os.path.join(HERE, 'run', 'missing_mode',
                                                  'sigma_base_ablation.json'))
    args = ap.parse_args()
    shapes = load_shapes()
    tags = ('bad', 'good') if args.shape == 'both' else (args.shape,)
    out = {}
    for tag in tags:
        if args.verify:
            verify(tag, shapes[tag])
        else:
            run_shape(tag, shapes[tag], out)
    if args.verify:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
