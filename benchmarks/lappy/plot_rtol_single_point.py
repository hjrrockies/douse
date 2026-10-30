"""Figure for `benchmarks/rtol_single_point.py`: sigma_1 at a non-eigenvalue, vs `rtol`.

    uv run python -m benchmarks.plot_rtol_single_point [--out rtol_single_point.png]

TOP ROW, one panel per domain: `sigma_1(lam; rtol)` for eight well-separated non-eigenvalues
(thin grey) and their median (heavy). The horizontal line is the target plateau that makes every
well at least one scan step wide; the shaded band above it is where a scan cannot step over a
mode. The vertical line is `lappy`'s default truncation.

BOTTOM ROW: the singular value spectrum of `R` at the median `lam` -- the object `rtol` actually
thresholds -- with the same default marked, so the count of discarded directions can be read off.

The point of putting the eight curves on the same axes is to answer whether ONE of them would
have done: if they lie on top of each other, a single tension evaluation is as good as forty.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'run', 'missing_mode', 'rtol_single_point.json')
LAPPY_DEFAULT = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'run', 'missing_mode',
                                                  'rtol_single_point.png'))
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()
    data = json.load(open(SRC))

    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(4.3*n, 6.6),
                             gridspec_kw=dict(height_ratios=[2.0, 1.0]))
    axes = np.atleast_2d(axes)

    for j, dd in enumerate(data):
        ax = axes[0][j]
        rt = np.array(dd['rtols'])
        S = np.array(dd['sigma'])
        tgt = dd['target']

        ax.axhspan(tgt, 1e2, color='#2A9D8F', alpha=0.10, lw=0)
        for i, lm in enumerate(dd['lams']):
            ax.loglog(rt, S[i], color='0.72', lw=0.9)
        ax.loglog(rt, np.median(S, axis=0), color='#C0392B', lw=2.0,
                  label=r'median of 8 $\lambda$')
        ax.axhline(tgt, color='#2A9D8F', lw=1.4, ls='--',
                   label=f'target {tgt:.1e}')
        ax.axvline(LAPPY_DEFAULT, color='#1F3D6B', lw=1.2, ls=':',
                   label=r'lappy default $10^{-12}$')
        ax.set_title(f"{dd['name']}\n"
                     f"$|\\Omega|/|\\partial\\Omega|$ = {dd['A_P']:.4f},  "
                     f"$\\Delta k$ = {dd['dk']:.4f}", fontsize=9)
        ax.set_xlabel('rtol  (absolute cutoff on singular values of $R$)')
        if j == 0:
            ax.set_ylabel(r'$\sigma_1$ at a non-eigenvalue')
        ax.set_ylim(5e-4, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc='upper left')

        # how far apart are the eight curves? that is the single-point question
        spread = float(np.max(S.max(axis=0)/S.min(axis=0)))
        ax.annotate(f'curves agree to {spread:.1f}x',
                    xy=(0.97, 0.04), xycoords='axes fraction', ha='right', fontsize=7.5,
                    color='0.35', bbox=dict(fc='white', ec='0.8', alpha=0.9))

        axb = axes[1][j]
        spec = np.array(dd['r_spectrum'])
        axb.semilogy(np.arange(1, len(spec) + 1), spec, '.', ms=3, color='#1F3D6B')
        axb.axhline(LAPPY_DEFAULT, color='#1F3D6B', lw=1.2, ls=':')
        kept = int((spec >= LAPPY_DEFAULT).sum())
        axb.annotate(f'{kept} of {len(spec)} kept\nat $10^{{-12}}$',
                     xy=(0.97, 0.80), xycoords='axes fraction', ha='right', va='top',
                     fontsize=7.5, color='#1F3D6B')
        axb.set_xlabel('index')
        if j == 0:
            axb.set_ylabel('singular values of $R$')
        axb.set_ylim(1e-17, 1e2)
        axb.grid(alpha=0.25)

    fig.suptitle('Tension at a non-eigenvalue as a function of the rank truncation\n'
                 'shaded band = wells at least one scan step wide (findable)', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
