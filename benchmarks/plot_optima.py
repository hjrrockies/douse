"""The optimal shapes found so far, as a grid of `k` (rows) against `N` (columns).

    uv run python -m benchmarks.plot_optima [--out optima.png] [--include-pre-fix]

Reads the multi-start ledgers and draws the best CERTIFIED shape in each `(N, k)` cell, at true
relative scale -- every polygon here has perimeter 4, so the panels are directly comparable and
a wider shape really is wider.

WHAT EACH PANEL SAYS. The title carries `lam_k/lam_1` and, underneath, the evidence: how many
independent starts certified that shape out of how many ran, and the spread between them. A
filled outline means the cell has a certificate; a dashed grey one means the best run in that
cell stopped without one, and the number is a LOWER BOUND rather than an answer.

`--include-pre-fix` folds in `run/ratio_multistart/pre_plateau_fix/`, the ledgers produced before
the resolution floor and the plateau stop. Those rows have no `resolution` field and mostly ended
on a radius collapse, so they are drawn as uncertified regardless of what they reported -- they
are there to fill panels the current grid has not reached yet, and they are marked `pre-fix` so
nobody quotes them as results.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from douse import VertexPolygon, spectral_guard

PERIM = 4.0
LEDGERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run', 'ratio_multistart')
# THE GATES ARE IMPORTED, NOT RESTATED. Three files now read these ledgers and a guard
# copied into each is a guard that drifts apart across them.
try:
    from .summarize_grid import PPW, row_deficit
except ImportError:                                                   # pragma: no cover
    from summarize_grid import PPW, row_deficit


def canonical(v):
    """Rotate so the longest edge lies horizontally along the bottom, interior above.

    WITHOUT THIS THE FIGURE MISLEADS. Shapes come out of the optimizer in whatever orientation the
    gauge fixing left them, and two panels can look like the same shape at a glance while being a
    symmetric trapezoid (65.4/65.4/114.6/114.6) and a scalene quadrilateral (69.2/80.9/91.5/118.4)
    -- which is exactly what happened on the first version of this plot.

    This is a PLOTTING convention only. `polygon.param_vector`'s docstring warns that picking the
    longest side is a map with a branch cut where sides tie, which is a genuine problem for the
    parametrization; here a tie just picks one of several equivalent orientations of the same
    picture, which is harmless.
    """
    v = np.asarray(v) - np.asarray(v).mean()
    e = np.roll(v, -1) - v
    i = int(np.argmax(np.abs(e)))
    v = v*np.exp(-1j*np.angle(e[i]))                  # longest edge -> along +x
    v = v - 1j*v[i].imag                              # that edge -> y = 0
    # signed area > 0 puts the interior above the edge rather than below it
    if np.sum((v.real*np.roll(v.imag, -1) - np.roll(v.real, -1)*v.imag)) < 0:
        v = np.conj(v)
    return v - v.mean().real


def _load(path, pre_fix=False):
    """`(usable rows, starts attempted)`.

    THE DENOMINATOR IS ATTEMPTED STARTS, and it used to be usable ones. A crashed row was
    dropped here before anything counted it, so this file reported N=6 `lam_2/lam_1` as 3/7
    where `summarize_grid` reported 3/8 for the same cell -- neither wrong, both defensible, and
    the same cell reading two ways will cost somebody an hour. A start that crashed was still a
    start, and hiding it flatters the cell.
    """
    rows, attempted = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        attempted += 1
        r = json.loads(line)
        if not r.get('ok', True):
            continue
        r['pre_fix'] = pre_fix
        rows.append(r)
    return rows, attempted


def collect(include_pre_fix, sigma_max=1e-6, res_max=1e-4, convex=False):
    """`{(N, k): summary}` -- the best certified run per cell, else the best clean one.

    ONE FAMILY PER FIGURE. `--convex` cells live in `N<N>_k<k>_convex.jsonl` and answer a
    different question -- the maximum over CONVEX N-gons -- so they are never mixed into the
    unconstrained panel grid, which would silently plot a constrained optimum as the answer.
    """
    cells = {}
    sources = [(os.path.join(LEDGERS, 'N*_k*_convex.jsonl' if convex else 'N*_k*.jsonl'),
                False)]
    if include_pre_fix:
        sources.append((os.path.join(LEDGERS, 'pre_plateau_fix', 'N*_k*.jsonl'), True))
    for pattern, pre in sources:
        for path in sorted(glob.glob(pattern)):
            if not convex and path.endswith('_convex.jsonl'):
                continue
            rows, attempted = _load(path, pre)
            if not rows:
                continue
            N, k = rows[0]['N'], rows[0]['k']
            if (N, k) in cells and not cells[(N, k)]['pre_fix']:
                continue                      # a post-fix cell always wins over a pre-fix one
            clean = [r for r in rows if r['worst_sigma'] <= sigma_max]
            # PPW IS A THEOREM, SO A k=1 ROW ABOVE IT IS NOT A SHAPE WORTH DRAWING. The N=6
            # `lam_2/lam_1` cell produced 2.6203171651 from a run whose `worst_sigma` was a
            # pristine 1e-09: it had SKIPPED lam_2 entirely and optimized lam_3/lam_1 while
            # believing otherwise. `worst_sigma` cannot see a missed mode -- it measures how well
            # the basis resolves what it FOUND -- so the sigma filter above passes it untouched.
            # That run happened not to certify, so the `good` pool excluded it by luck rather
            # than by a check. In a cell where nothing certifies, `pool` falls back to `clean`
            # and the impossible shape would be drawn as the answer.
            impossible = [r for r in clean if r['k'] == 1 and r['fun'] > PPW + 1e-9]
            clean = [r for r in clean if r not in impossible]
            # AND THE GENERAL FORM OF THE SAME FAILURE, which PPW only catches at k=1: a tracked
            # set that is not consecutive. Recomputed from `x` and `lams` so it applies to rows
            # written before the guard existed. See `douse.spectral_guard`.
            def _skipped(r):
                d = row_deficit(r)
                return np.isfinite(d) and d > spectral_guard.DEFICIT_TOL
            n_skipped = sum(_skipped(r) for r in clean)
            clean = [r for r in clean if not _skipped(r)]
            if n_skipped:
                print(f'  N={N} k={k}: dropped {n_skipped} run(s) whose tracked set is not '
                      f'consecutive (Weyl deficit > {spectral_guard.DEFICIT_TOL})')
            if impossible:
                print(f'  N={N} k={k}: dropped {len(impossible)} run(s) above the PPW bound '
                      f'{PPW:.7f} (max {max(r["fun"] for r in impossible):.7f}) -- a missed mode')
            if not clean:
                continue
            # THE CERTIFICATE IS THE STOP, not a threshold on the resolution -- see the note
            # in `summarize_grid.summarize`. Gating on the plateau height certifies by the
            # shape's noise floor rather than by optimality, and it was rejecting 20 rows of
            # which 17 agreed with their cell's best to 1e-6.
            # pre-fix rows have no `resolution` and are never treated as certified
            good = [r for r in clean if (not pre) and r.get('success')
                    and np.isfinite(r.get('resolution', np.nan))]
            pool = good or clean
            best = max(pool, key=lambda r: r['fun'])
            agree = [r for r in pool if abs(r['fun'] - best['fun']) <= 1e-6*abs(best['fun'])]
            # A CELL IS CONTESTED WHEN AN EXCLUDED ROW CLAIMS A HIGHER VALUE AND THE REASON
            # IT WAS EXCLUDED IS ITSELF IN DOUBT. A missing mode INFLATES a ratio -- the lam_k
            # slot gets filled by a higher eigenvalue -- so "excluded and higher" is both the
            # signature of that failure and what a genuinely better basin looks like, and only
            # re-solving separates them.
            #
            # THE FIRST VERSION OF THIS FLAGGED THREE CELLS AND TWO WERE SETTLED, so the rule
            # excludes what is already decided:
            #   * above the PPW bound at k=1 is IMPOSSIBLE, not contested (N=6 lam_2/lam_1's
            #     2.6203171651, at a pristine sigma of 1.3e-09 -- the row that motivated the
            #     whole missed-mode guard);
            #   * a drifted basis is not a result (N=6 lam_3/lam_1's 3.6372 at sigma 6.3e-04);
            #   * a deficit far above threshold is a missing mode, not a marginal call. 1.497
            #     is; 0.534 and 0.558, against a threshold of 0.5 calibrated on N=4..6 at k<=5,
            #     are exactly where a false positive would sit.
            excluded = [r for r in rows if r not in clean]
            contested = sorted(
                (r['fun'], r['start']) for r in excluded
                if r['fun'] > best['fun']*(1 + 1e-9)
                and r.get('worst_sigma', 1.0) <= sigma_max
                and not (r['k'] == 1 and r['fun'] > PPW + 1e-9)
                and row_deficit(r) <= 2*spectral_guard.DEFICIT_TOL)
            cells[(N, k)] = dict(best=best, n_agree=len(agree), n_total=attempted,
                                 n_crashed=attempted - len(rows), contested=contested,
                                 certified=bool(good), pre_fix=pre,
                                 spread=(float(np.ptp([r['fun'] for r in agree]))
                                         if len(agree) > 1 else float('nan')))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='optima.png')
    ap.add_argument('--include-pre-fix', action='store_true')
    ap.add_argument('--convex', action='store_true',
                    help='plot the CONVEX cells (N<N>_k<k>_convex.jsonl) instead. The two '
                         'families are never drawn on the same grid.')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    cells = collect(args.include_pre_fix, convex=args.convex)
    if not cells:
        print('no cells found')
        return
    Ns = sorted({N for N, _ in cells})
    ks = sorted({k for _, k in cells})
    fig, axes = plt.subplots(len(ks), len(Ns), squeeze=False,
                             figsize=(2.3*len(Ns) + 0.7, 2.5*len(ks) + 0.7))

    # ONE SCALE FOR EVERY PANEL, so the shapes are comparable across the whole figure -- but
    # each panel is CENTRED ON ITS OWN SHAPE. `canonical` centres in x and puts the longest edge
    # on y = 0, which leaves every polygon sitting entirely in the upper half plane; measuring the
    # span as `max|v|` about the origin then doubles it, and every shape is drawn at half the size
    # the figure has room for. The half-width below is the largest half-extent any shape needs
    # about its OWN bounding-box centre, so the scale is still common and the shapes fill the grid.
    shapes = {key: canonical(VertexPolygon(c['best']['N'], perim=PERIM)
                             .vertices(np.array(c['best']['x'])))
              for key, c in cells.items()}
    centre = {key: (0.5*(v.real.max() + v.real.min()), 0.5*(v.imag.max() + v.imag.min()))
              for key, v in shapes.items()}
    span = max(max(float(np.abs(v.real - centre[key][0]).max()),
                   float(np.abs(v.imag - centre[key][1]).max()))
               for key, v in shapes.items())
    span *= 1.12

    for i, k in enumerate(ks):
        for j, N in enumerate(Ns):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            c = cells.get((N, k))
            cx, cy = centre.get((N, k), (0.0, 0.0))
            ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
            if c is None:
                ax.text(0, 0, '—', ha='center', va='center', color='0.75', fontsize=18)
                for sp in ax.spines.values():
                    sp.set_color('0.9')
                continue
            b = c['best']
            v = shapes[(N, k)]
            xy = np.column_stack([v.real, v.imag])
            xy = np.vstack([xy, xy[:1]])
            contested = c.get('contested') or []
            if contested:
                # drawn, but visibly NOT settled
                ax.fill(xy[:, 0], xy[:, 1], color='#C8791E', alpha=0.22, zorder=1)
                ax.plot(xy[:, 0], xy[:, 1], color='#8A4B00', lw=1.8, ls=(0, (5, 2)), zorder=2)
                edge = '#8A4B00'
            elif c['certified']:
                ax.fill(xy[:, 0], xy[:, 1], color='#4C72B0', alpha=0.30, zorder=1)
                ax.plot(xy[:, 0], xy[:, 1], color='#1F3D6B', lw=1.8, zorder=2)
                edge = '#1F3D6B'
            else:
                ax.plot(xy[:, 0], xy[:, 1], color='0.45', lw=1.4, ls='--', zorder=2)
                edge = '0.45'
            ax.plot(xy[:-1, 0], xy[:-1, 1], 'o', ms=2.6, color=edge, zorder=3)
            if contested:
                for sp in ax.spines.values():
                    sp.set_color('#C8791E'); sp.set_linewidth(1.6)

            spread = ('' if not np.isfinite(c['spread']) else f", spread {c['spread']:.0e}")
            crashed = (f", {c['n_crashed']} crashed" if c.get('n_crashed') else '')
            note = (f"{c['n_agree']}/{c['n_total']} certified{spread}{crashed}"
                    if c['certified']
                    else (f"lower bound ({c['n_total']} run(s)){crashed}"
                          + (', pre-fix' if c['pre_fix'] else '')))
            if contested:
                top = contested[-1][0]
                note += (f"\nCONTESTED: {len(contested)} excluded run(s) up to {top:.7f}")
            ax.set_title(f"$\\lambda_{{{k+1}}}/\\lambda_1$ = {b['fun']:.7f}\n{note}",
                         fontsize=8.5, pad=4,
                         color='black' if c['certified'] else '0.35')

    for j, N in enumerate(Ns):
        axes[0][j].annotate(f'N = {N}', xy=(0.5, 1.42), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=13, fontweight='bold')
    for i, k in enumerate(ks):
        axes[i][0].annotate(f'k = {k+1}', xy=(-0.16, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=13, fontweight='bold',
                            rotation=90)

    fig.suptitle('Maximizers of $\\lambda_k/\\lambda_1$ over N-gons of perimeter 4\n'
                 'blue = first-order certificate;  dashed grey = lower bound only;  '
                 'amber = contested, an excluded run claims more;  all panels to the same scale',
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0.015, 0.0, 1, 0.965))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f'wrote {args.out}  ({len(cells)} cell(s): '
          f'{sum(c["certified"] for c in cells.values())} certified)')


if __name__ == '__main__':
    main()
