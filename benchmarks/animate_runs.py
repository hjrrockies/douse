"""Animate the optimization runs: shape on the left, spectrum on the right.

    .venv/bin/python -m benchmarks.animate_runs [name ...]

Follows the format of `lappy/jupyter_notebooks/equi_tri1.mp4` -- the current shape drawn against
the known answer in grey dashes, and the eigenvalues traced against their exact values in black
dashes -- with a third panel added, because these runs converge through fourteen orders of
magnitude and a linear axis shows none of it.

Three runs, and each is showing something different:

* `min_lam1_triangle`  -- converges to the equilateral triangle (Polya-Szego).
* `min_lam1_quad`      -- converges to the square. Same theorem, one more dimension.
* `min_lam2_quad`      -- converges to a RECTANGLE, not the square: `lam_2` is minimized at
  `L = 2*4^(1/3)/(1+4^(1/3)) = 1.2270`. The right panel is the point of this one -- `lam_2` and
  `lam_3` stay far apart the whole way, which is the measurement that said a cluster optimizer
  is not yet needed.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from douse import (VertexPolygon, param_vector, SpectrumEvaluator, ObjectiveFunction,
                   EigenvalueObjective, descend)

PERIM = 4.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animations')

_c = 4**(1/3)
_L = 2*_c/(1 + _c)

RUNS = {
    'min_lam1_triangle': dict(
        n=3, k=0, n_eigs=1, max_iter=60,
        start=np.array([0.0, 1.5, 0.55 + 1.1j]),
        target=np.array([0.0, 4/3, 2/3 + (2/np.sqrt(3))*1j/1.0]),
        exact=3*np.pi**2,
        title=r'minimize $\lambda_1$  $\to$  equilateral triangle'),
    'min_lam1_quad': dict(
        n=4, k=0, n_eigs=1, max_iter=150,
        start=np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j]),
        target=np.array([0.0, 1.0, 1.0 + 1j, 1j]),
        exact=2*np.pi**2,
        title=r'minimize $\lambda_1$  $\to$  square'),
    'min_lam2_quad': dict(
        n=4, k=1, n_eigs=3, max_iter=90,
        start=np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j]),
        target=np.array([0.0, _L, _L + (2 - _L)*1j, (2 - _L)*1j]),
        exact=np.pi**2*(4/_L**2 + 1/(2 - _L)**2),
        title=r'minimize $\lambda_2$  $\to$  rectangle, not the square'),
}


def _equilateral(perim):
    a = perim/3
    return np.array([0.0, a, a/2 + a*np.sqrt(3)/2*1j])


def run(cfg):
    """Descend, recording the shape and spectrum at every accepted iterate."""
    fam = VertexPolygon(cfg['n'], perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=cfg['n_eigs'], warn_gap=0.0)
    obj = ObjectiveFunction(ev, EigenvalueObjective(cfg['k']))
    p0 = param_vector(cfg['start'], PERIM)

    shapes, spectra = [fam.vertices(p0)], []

    def cb(p, val, grad):
        shapes.append(fam.vertices(p))
        spectra.append(np.array(ev._last[1]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        val0, _ = obj(p0)
        spectra.append(np.array(ev._last[1]))
        res = descend(obj, p0, max_iter=cfg['max_iter'], gtol=1e-7, callback=cb)
    return fam, shapes, np.array(spectra), res


def _closed(v):
    return np.append(v, v[0])


def animate(name, cfg, fam, shapes, spectra, res, fps=8, hold=12):
    """Three panels: the shape, the spectrum, and the error on a log axis.

    `hold` repeats the final frame so the converged shape stays on screen instead of flashing
    past -- most of the descent happens in the first handful of iterations and the rest is the
    last few digits.
    """
    target = _equilateral(PERIM) if cfg['n'] == 3 else cfg['target']
    # put the target in the same gauge as the iterates, so the two share one axis
    target = fam.vertices(param_vector(target, PERIM))

    fig, (ax_s, ax_e, ax_c) = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    k = cfg['k']

    allv = np.concatenate([np.concatenate(shapes), target])
    pad = 0.08*max(np.ptp(allv.real), np.ptp(allv.imag))
    ax_s.set_xlim(allv.real.min() - pad, allv.real.max() + pad)
    ax_s.set_ylim(allv.imag.min() - pad, allv.imag.max() + pad)
    ax_s.set_aspect('equal')
    tv = _closed(target)
    ax_s.plot(tv.real, tv.imag, '--', color='0.6', lw=1.5, label='exact minimizer')
    (line_s,) = ax_s.plot([], [], '-', color='C0', lw=2)
    ax_s.legend(loc='upper right', fontsize=8, frameon=False)

    n_it = len(spectra)
    ax_e.set_xlim(-0.5, n_it - 0.5)
    lo, hi = spectra.min(), spectra.max()
    ax_e.set_ylim(lo - 0.06*(hi - lo), hi + 0.10*(hi - lo))
    ax_e.set_xlabel('iteration')
    ax_e.set_title('Eigenvalues')
    ax_e.axhline(cfg['exact'], color='k', ls='--', lw=1.5)
    ax_e.annotate(rf"exact $\lambda_{{{k+1}}}$ = {cfg['exact']:.6f}",
                  xy=(0.98, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=9)
    lines_e = [ax_e.plot([], [], lw=1.5, marker='.', ms=3, color=f'C{j}',
                         label=rf'$\lambda_{{{j+1}}}$')[0] for j in range(spectra.shape[1])]
    if spectra.shape[1] > 1:
        ax_e.legend(loc='center right', fontsize=8, frameon=False)

    err = np.abs(spectra[:, k] - cfg['exact'])
    floor = max(err[err > 0].min() if np.any(err > 0) else 1e-16, 1e-16)
    ax_c.set_xlim(-0.5, n_it - 0.5)
    ax_c.set_ylim(0.3*floor, 3*err.max())
    ax_c.set_yscale('log')
    ax_c.set_xlabel('iteration')
    ax_c.set_title(rf'$|\lambda_{{{k+1}}} - $ exact$|$')
    (line_c,) = ax_c.plot([], [], lw=1.5, marker='.', ms=3, color='C3')

    fig.suptitle(cfg['title'], fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(f):
        i = min(f, n_it - 1)
        v = _closed(shapes[i])
        line_s.set_data(v.real, v.imag)
        ax_s.set_title(f'iteration {i}', fontsize=11)
        xs = np.arange(i + 1)
        for j, ln in enumerate(lines_e):
            ln.set_data(xs, spectra[:i + 1, j])
        line_c.set_data(xs, np.maximum(err[:i + 1], 0.3*floor))
        return [line_s, line_c, *lines_e]

    anim = FuncAnimation(fig, update, frames=n_it + hold, blit=False)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{name}.mp4')
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    return path, n_it


def main(names):
    for name in names:
        cfg = RUNS[name]
        print(f'{name}: running...', flush=True)
        fam, shapes, spectra, res = run(cfg)
        path, nf = animate(name, cfg, fam, shapes, spectra, res)
        err = res.fun - cfg['exact']
        print(f'  {res.nit} iterations, lam_{cfg["k"]+1} - exact = {err:+.3e}')
        print(f'  wrote {path}  ({nf} frames)')


if __name__ == '__main__':
    main(sys.argv[1:] or list(RUNS))
