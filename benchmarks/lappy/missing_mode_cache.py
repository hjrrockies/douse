"""Everything `jupyter_notebooks/missing_mode.ipynb` plots, computed once and cached.

    uv run python -m benchmarks.missing_mode_cache [--recompute] [--good-from-repro]

THE NOTEBOOK SHOULD NOT HOLD THE MEASUREMENTS. An MPS solve at these shapes costs 3-50 s and the
FEM ground truth needs gmsh, so a notebook that recomputes on every open is a notebook nobody
re-opens. This writes one `.npz` and the notebook is then pure plotting, ~10 s warm.

WHAT IS BEING MEASURED. One shape from `run/ratio_multistart/N3_k2.jsonl` start 4 -- a 25/133/22
degree triangle -- at which a FRESH full solve returns four eigenvalues at a tension of 2.8e-14,
with no exception and no warning, while three modes between them are absent. The refusal record
for that row has `plan_at == p` and `from_full_solve=True`, so the miss is a property of the shape
and the search, not of a basis frozen somewhere else.

The cache holds, for the failing shape and for a partner shape the same run visited:
  * what `lappy` returned, its tension, and the two completeness statistics;
  * the FEM spectrum, which is a ONE-SIDED bound and therefore a proof rather than a heuristic;
  * a dense sigma(lambda) sweep, plus fine sweeps around each true mode;
  * the scan the solver actually runs -- its window, its grid, what it accepted and what it
    rejected -- which is what separates "the grid stepped over the well" from "the well was found
    and its tension missed `ttol`".
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

import lappy
from lappy import asymp, bounds, mps
from lappy import evp as _evp

from douse import VertexPolygon, spectral_guard
from douse.evaluate import SpectrumEvaluator, ShapeRefusal
from douse import degeneracy as dg

from .completeness_detectors import fem_spectrum, fem_deficit

PERIM = 4.0
N, K = 3, 2
HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, 'run', 'ratio_multistart', 'N3_k2.jsonl')
REPRO = os.path.join(HERE, 'diagnostics', 'repro_N3_k2_s4.jsonl')
OUTDIR = os.path.join(HERE, 'run', 'missing_mode')
OUT = os.path.join(OUTDIR, 'cache.npz')

FAM = VertexPolygon(N, perim=PERIM)
FEM_HFRAC = 150                    # 2704 dofs here; resolves comfortably through lam ~ 440
SWEEP_LO, SWEEP_HI = 40.0, 460.0
SWEEP_COARSE = 0.35                # baseline resolution away from any mode
SWEEP_FINE, SWEEP_HALF = 0.01, 3.0  # inside a window around each true mode
ZOOM_HALF, ZOOM_N = 3.0, 601


def sweep_grid(fem, lo=SWEEP_LO, hi=SWEEP_HI):
    """A COMPOSITE grid: coarse everywhere, fine near every true mode.

    A uniform grid cannot draw this picture. The wells here are 0.1 to 0.3 wide, so a uniform
    sweep fine enough to render them costs ~20k points, and one cheap enough to afford (0.35)
    aliases every well it is supposed to show -- the first version of this figure plotted a
    baseline with a few ragged notches and hid the very structure the notebook is about.

    `SWEEP_HALF = 3.0` rather than something tighter because the fine windows are centred on the
    FEM values, which are UPPER bounds: at lam ~ 383 the P1 estimate sits 1.5 above the true mode,
    and a narrower window would centre the fine sampling next to the well instead of on it.
    """
    parts = [np.arange(lo, hi, SWEEP_COARSE)]
    for c in np.asarray(fem, float):
        if lo < c < hi:
            parts.append(np.arange(max(lo, c - SWEEP_HALF), min(hi, c + SWEEP_HALF), SWEEP_FINE))
    return np.unique(np.concatenate(parts))


def evaluator(p, n_eigs=K + 2):
    """The diagnostic configuration used across this repo: every guard that would MASK the
    failure is switched off, and the basis is planned at the shape itself so staleness is not a
    confound. A FRESH evaluator per measurement -- its statistics are lifetime running maxima."""
    return SpectrumEvaluator(FAM, n_eigs=n_eigs, warn_gap=0.0, degenerate_tol=0.0,
                             lam_max_window=max(8, 2*K + 4), replan_tol=None,
                             plan_at=np.asarray(p, float), enforce_weyl=False)


def geom(p):
    v = FAM.vertices(np.asarray(p, float))
    return (np.sort(dg.edge_lengths(v)), np.sort(np.degrees(dg.interior_angles(v))),
            float(FAM(np.asarray(p, float)).area))


def solve(p, n_eigs=K + 2):
    """`dict` of what `lappy` says at `p`, plus what is actually true there."""
    p = np.asarray(p, float)
    out = dict(p=p.tolist())
    ev = evaluator(p, n_eigs)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lams, _ = ev.eigenvalues(p)
        lams = np.asarray(lams, float)
        out.update(lams=lams.tolist(), refused=False, error='')
    except ShapeRefusal as exc:
        # A REFUSAL IS A DATA POINT, not a failure of the harness: it is the LOUD version of the
        # same defect, and the contrast with the silent one is half the story.
        out.update(lams=[], refused=True, error=f'{type(exc).__name__}: {exc}')
    out['seconds'] = round(time.perf_counter() - t0, 1)

    dom = FAM(p)
    if not out['refused']:
        lams = np.asarray(out['lams'], float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out['sigma'] = float(ev.sigma_at(p, lams[0]))
        out['weyl_deficit'] = float(spectral_guard.weyl_deficit(dom, lams))
        lam_top = 1.15*float(lams[-1])
    else:
        out['sigma'] = float('nan')
        out['weyl_deficit'] = float('nan')
        lam_top = 460.0
    fem, ndof = fem_spectrum(dom, FEM_HFRAC, lam_top)
    out['fem'] = np.asarray(fem, float).tolist()
    out['n_dof'] = int(ndof)
    out['fem_deficit'] = (int(fem_deficit(np.asarray(fem, float), np.asarray(out['lams'], float)))
                          if not out['refused'] else -1)
    e, a, area = geom(p)
    out.update(edges=e.tolist(), angles=a.tolist(), area=area)
    return out


def sweep(p, grid):
    """`(sigma_1, sigma_2)` along `grid`.

    `ev.solver(p)` costs ~20 ms -- the plan is nearly free, and it is `eigenvalues` that is slow.
    NOT `solver.sigma(grid)`: that is `tensions(grid)[0]`, the whole tension spectrum at the FIRST
    lambda, which looks like a plausible curve and is not one.
    """
    ev = evaluator(p)
    _, solver = ev.solver(np.asarray(p, float))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = solver.tensions(np.asarray(grid, float), n_workers=mps.n_workers_default)
    return (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float))


def scan_window(p, k_search=K + 2):
    """`(a, b, n_pts, nodes)` -- the window and grid `Eigenproblem.solve` lays down at `p`.

    Confirmed against `lappy/evp.py:334-360`: for Dirichlet `k_search = k`, `m = k_search + 1`,
    `a = faber_krahn(area)*(1 - _WINDOW_PAD)`, `b = weyl_est(m)`, `n_pts = ppl*m` with `ppl = 20`.
    """
    dom = FAM(np.asarray(p, float))
    a = float(bounds.faber_krahn(area=dom.area)*(1 - _evp._WINDOW_PAD))
    b = float(asymp.weyl_est(k_search + 1, dom))
    n_pts = 20*(k_search + 1)
    return a, b, n_pts, np.asarray(mps.make_lamgrid(a, b, n_pts), float)


def sigma_on_scan_grid(p, k_search=K + 2):
    """`(nodes, sigma_1)` at exactly the points the search evaluates.

    MEASURED, NOT INTERPOLATED. The figure built on this claims to show what the search sees, and
    reading it off a dense sweep would be a derived curve presented as data -- particularly
    misleading here, where the function between samples is precisely what is in question.
    """
    _, _, _, nodes = scan_window(p, k_search)
    s1, _ = sweep(p, nodes)
    return nodes, s1


def scan(p, k_search=K + 2):
    """The scan `Eigenproblem.solve` actually runs at `p`, with its rejects kept.

    `evp._solve_dir_neu` builds the window from the Faber--Krahn bound up to `weyl_est(k+1)` and
    lays `ppl*m` points across it; a polished minimum becomes an eigenvalue only if its tension is
    at or below `ttol`. `return_rejected=True` is the flag that distinguishes the two ways this
    loses a mode, and `Eigenproblem.solve` does not surface it.
    """
    p = np.asarray(p, float)
    ev = evaluator(p)
    dom, solver = ev.solver(p)
    a = float(bounds.faber_krahn(area=dom.area)*(1 - _evp._WINDOW_PAD))
    b = float(asymp.weyl_est(k_search + 1, dom))
    ppl, m = 20, k_search + 1
    n_pts = ppl*m
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eigs, mults, fevals, rejected = solver.solve_interval(a, b, n_pts, return_rejected=True)
    rej = np.asarray(rejected, float).reshape(-1, 2)
    return dict(a=a, b=b, n_pts=int(n_pts), ppl=ppl,
                grid=np.asarray(mps.make_lamgrid(a, b, n_pts), float).tolist(),
                eigs=np.asarray(eigs, float).tolist(),
                mults=np.asarray(mults, int).tolist(),
                fevals=int(fevals),
                rejected=rej.tolist(), ttol=float(mps.ttol_default))


def good_shape_from_repro():
    """`(p_prev, p_refused, n_spurious)` from the instrumented re-run, or `(None, None, 0)`.

    THE PAIR THE STORY IS ABOUT: the last domain at which a spectrum came back cleanly, and the
    very next one the run evaluated, at which the guard refused. `p_prev` rather than the last
    accepted ITERATE, because the optimizer evaluates trial shapes between iterates and the
    immediately preceding evaluation is the tighter, more honest pairing.

    `n_spurious` counts the LOUD refusals the walk survived on the way -- worth reporting, since
    it shows the run passing through a region where the solver complains before reaching the one
    where it does not.
    """
    if not os.path.exists(REPRO):
        return None, None, 0
    n_spurious = 0
    for line in open(REPRO):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('kind') == 'spurious_refusal':
            n_spurious += 1
        elif r.get('kind') == 'refusal':
            return r.get('p_prev'), r.get('p'), n_spurious
    return None, None, n_spurious


def is_consecutive(lams, fem, rtol=0.03):
    """Is `lams` really `lam_1 .. lam_K`? Checked INDEX BY INDEX against the FEM spectrum.

    A count is not enough, and this function exists because a count let the wrong shape through.
    `fem_deficit` compares how many values lie below each mid-gap cut, so a DUPLICATED root --
    the same eigenvalue returned twice, a documented `lappy` failure -- adds a value where one is
    missing and the deficit reads a clean 0. The partner shape chosen that way returned
    `[88.0218, 88.0219, 141.1603, 190.2552]`: `lam_1` twice, which is impossible for a connected
    domain, with the true `lam_4` dropped off the end.

    Comparing position by position catches both directions at once. P1 elements give an upper
    bound, so each returned value should sit a little BELOW its FEM partner -- measured here at
    0.08% to 0.25% -- and `rtol=0.03` is loose enough for a coarse mesh while far tighter than the
    spacing between neighbouring modes.
    """
    lams, fem = np.asarray(lams, float), np.asarray(fem, float)
    if len(lams) == 0 or len(lams) > len(fem):
        return False
    return bool(np.all(np.abs(lams - fem[:len(lams)]) <= rtol*fem[:len(lams)]))


def nearest_clean_partner(p_bad, p_far, n_steps=6):
    """Walk from `p_far` toward `p_bad` and bracket where the spectrum stops being complete.

    Returns `(p_clean, t_clean, t_bad, trail)`: the closest shape to the failing one that still
    solves with a FEM deficit of zero, and the bracket the switch falls in.

    WHY THIS EXISTS. The intended partner was the literal preceding iterate, recovered by
    re-running the start with every call logged. That re-run does not reproduce the failure: it
    diverges at the first spurious refusal -- where the optimizer shrinks its radius, and float
    non-determinism decides by how much -- and then converges cleanly to the cell's answer in 153
    iterations with a worst deficit of 0.19. So the literal predecessor is not recoverable, and
    pairing the failing shape with the run's distant endpoint (0.72 away) would not support the
    claim that the failure switches on over a short walk.

    A bisection on the segment between two shapes the run DID visit does support it, and is
    honest about being a constructed path rather than the optimizer's own steps.
    """
    p_bad, p_far = np.asarray(p_bad, float), np.asarray(p_far, float)
    lo, hi = 0.0, 1.0                      # t=0 is p_far (assumed clean), t=1 is p_bad (fails)
    best, trail = None, []
    for _ in range(n_steps):
        t = 0.5*(lo + hi)
        q = (1 - t)*p_far + t*p_bad
        d = solve(q)
        clean = (not d['refused']) and is_consecutive(d['lams'], d['fem'])
        trail.append(dict(t=float(t), clean=bool(clean), refused=bool(d['refused']),
                          fem_deficit=int(d['fem_deficit']),
                          weyl=float(d['weyl_deficit']), lams=list(d['lams']), p=q.tolist()))
        print(f'    t={t:.4f}  '
              f'{"REFUSED" if d["refused"] else ("consecutive" if clean else "NOT consecutive")}  '
              f'fem_deficit={d["fem_deficit"]}  weyl={d["weyl_deficit"]:+.3f}  '
              f'lams={np.round(d["lams"], 3).tolist()}')
        if clean:
            lo, best = t, q
        else:
            hi = t
    return (p_far if best is None else best), lo, hi, trail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--recompute', action='store_true',
                    help='ignore an existing cache and rebuild every entry')
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.recompute:
        print(f'{args.out} exists; pass --recompute to rebuild')
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    row = None
    for line in open(LEDGER):
        r = json.loads(line)
        if r.get('start') == 4 and r.get('refusals'):
            row = r
    if row is None:
        raise SystemExit('no start-4 row with a recorded refusal in the ledger')
    ref = row['refusals'][0]
    p_bad = np.asarray(ref['p'], float)
    assert np.allclose(ref['plan_at'], p_bad), 'plan_at != p: this refusal IS a stale basis'

    p_good, p_bad_repro, n_spurious = good_shape_from_repro()
    if p_good is not None:
        same = np.allclose(np.asarray(p_bad_repro, float), p_bad, atol=1e-6)
        partner = ('the immediately preceding domain of the run, from the instrumented re-run'
                   + ('' if same else ' (which refused at a nearby shape, not the ledger\'s)'))
        if not same:
            # THE PAIR MUST BE A PAIR. Pairing the re-run's predecessor with a DIFFERENT shape's
            # failure would be two unrelated measurements presented as one step.
            print(f'  NOTE: the re-run refused at {np.round(p_bad_repro, 6).tolist()}, not at the '
                  f"ledger's {np.round(p_bad, 6).tolist()}; using the re-run's own pair")
            p_bad = np.asarray(p_bad_repro, float)
        p_good = np.asarray(p_good, float)
        bracket = None
    else:
        print('  the re-run reached no deficit refusal; bracketing the switch-on instead ...')
        p_good, t_lo, t_hi, trail = nearest_clean_partner(p_bad, np.asarray(row['x'], float))
        bracket = dict(t_clean=t_lo, t_bad=t_hi, trail=trail,
                       p_far=np.asarray(row['x'], float).tolist())
        partner = (f'the closest shape to the failing one that still solves completely, found by '
                   f'bisection on the segment from the run\'s endpoint (switch-on bracketed to '
                   f't in [{t_lo:.4f}, {t_hi:.4f}])')
    print(f'bad     p = {np.round(p_bad, 6).tolist()}')
    print(f'good    p = {np.round(p_good, 6).tolist()}   [{partner}]')

    meta = dict(partner=partner, n_spurious_before=int(n_spurious), bracket=bracket,
                row_success=bool(row.get('success')),
                row_message=str(row.get('message')), row_nit=int(row.get('nit', -1)),
                row_worst_sigma=float(row.get('worst_sigma', float('nan'))),
                row_n_unresolved=int(row.get('n_unresolved', -1)),
                row_worst_weyl_deficit=float(row.get('worst_weyl_deficit', float('nan'))),
                refusal_from_full_solve=bool(ref.get('from_full_solve')),
                refusal_deficit=float(ref.get('deficit')),
                deficit_tol=float(spectral_guard.DEFICIT_TOL),
                fem_hfrac=FEM_HFRAC, lappy_version=getattr(lappy, '__version__', '?'),
                p_bad_repro=(None if p_bad_repro is None else list(map(float, p_bad_repro))))

    store = {}
    for tag, p in (('good', p_good), ('bad', p_bad)):
        print(f'  solving {tag} ...', flush=True)
        d = solve(p)
        print(f'    lams={np.round(d["lams"],4).tolist()} sigma={d["sigma"]:.2e} '
              f'weyl={d["weyl_deficit"]:+.3f} fem_def={d["fem_deficit"]} ({d["seconds"]}s)')
        meta[tag] = d
        store[f'p_{tag}'] = np.asarray(p, float)
        store[f'lams_{tag}'] = np.asarray(d['lams'], float)
        store[f'fem_{tag}'] = np.asarray(d['fem'], float)

        print(f'  sweeping {tag} ...', flush=True)
        grid = sweep_grid(d['fem'])
        t0 = time.perf_counter()
        s1, s2 = sweep(p, grid)
        store[f'sweep_{tag}'] = np.vstack([grid, s1, s2])
        print(f'    {len(grid)} points in {time.perf_counter() - t0:.0f}s')

        # what the SEARCH sees: the tension at its own grid nodes, measured
        nodes, snode = sigma_on_scan_grid(p)
        store[f'scan_sigma_{tag}'] = np.vstack([nodes, snode])
        a_, b_, n_, _ = scan_window(p)
        meta.setdefault('windows', {})[tag] = dict(a=a_, b=b_, n_pts=n_,
                                                   spacing=(b_ - a_)/n_)
        print(f'    scan grid: {n_} nodes on [{a_:.1f}, {b_:.1f}], '
              f'min sigma at a node {snode.min():.2e}')

    print('  scan at the failing shape ...', flush=True)
    sc = scan(p_bad)
    meta['scan'] = sc
    store['scan_grid'] = np.asarray(sc['grid'], float)
    store['scan_eigs'] = np.asarray(sc['eigs'], float)
    store['scan_rejected'] = np.asarray(sc['rejected'], float).reshape(-1, 2)
    print(f"    window [{sc['a']:.1f}, {sc['b']:.1f}] at {sc['n_pts']} pts "
          f"(spacing {(sc['b']-sc['a'])/sc['n_pts']:.3f}); accepted "
          f"{np.round(sc['eigs'],4).tolist()}; {len(sc['rejected'])} rejected")

    print('  fine sweeps around each true mode at the failing shape ...', flush=True)
    fem_bad = np.asarray(meta['bad']['fem'], float)
    centres = fem_bad[fem_bad <= SWEEP_HI - ZOOM_HALF]
    zg, zs = [], []
    for c in centres:
        g = np.linspace(c - ZOOM_HALF, c + ZOOM_HALF, ZOOM_N)
        s1, _ = sweep(p_bad, g)
        zg.append(g)
        zs.append(s1)
    store['zoom_centres'] = np.asarray(centres, float)
    store['zoom_grid'] = np.asarray(zg, float)
    store['zoom_sigma'] = np.asarray(zs, float)

    print('  the LOUD variant: same shape, n_eigs=6 ...', flush=True)
    meta['bad_n6'] = solve(p_bad, n_eigs=6)
    print(f"    {'refused: ' + meta['bad_n6']['error'][:90] if meta['bad_n6']['refused'] else 'did not refuse'}")

    store['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(args.out, **store)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1e3:.0f} kB)')


if __name__ == '__main__':
    main()
