"""Shared plumbing for the cluster-optimizer animations.

Each `animate_*.py` script owns one scenario -- a starting shape, a `k`, a target -- and calls
`run_and_record` once to drive `douse.cluster_opt.maximize_ratio` and write every accepted
iterate's shape and spectrum to `benchmarks/run/cluster_frames/<name>.npz`, then `render` to turn
that into an mp4. The two are split on purpose: `render` only ever reads the .npz, so a new color
scheme or a different fps costs nothing -- the eigensolves that dominate a run's wall clock are
not repeated to change how the video looks.

Layout is a 2x2 grid rather than the earlier three-panel-wide one, per request for something more
compact: shape top-left, eigenvalue trajectories top-right, ratio error bottom-left, spectral gap
bottom-right. The gap panel is new here and is the point of these particular runs -- it is what
`docs/todo.md` measures when it says a run "entered cluster mode", so points are colored by
`model['mode']` to show the switch happening rather than just its consequence.
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from douse import VertexPolygon, param_vector, SpectrumEvaluator
from douse.cluster_opt import maximize_ratio

PERIM = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'run', 'cluster_frames')
OUT_DIR = os.path.join(HERE, 'animations')


def data_path(name):
    return os.path.join(DATA_DIR, f'{name}.npz')


def _regular(n, perim=PERIM):
    return perim/n/(2*np.sin(np.pi/n))*np.exp(2j*np.pi*np.arange(n)/n)


def _perturbed(n, jitter, seed=0, perim=PERIM):
    rng = np.random.default_rng(seed)
    v = _regular(n, perim)
    return v + jitter*(rng.normal(size=n) + 1j*rng.normal(size=n))*np.abs(v).mean()


def _gap(lams):
    lams = np.sort(np.asarray(lams, dtype=float))
    return float(np.min(np.diff(lams)/lams[:-1])) if len(lams) > 1 else np.inf


def run_and_record(name, n, k, start, target, exact_ratio, n_eigs=3, max_iter=150, delta0=0.05,
                   seed=0, force=False):
    """Run `maximize_ratio`, recording every iterate's shape and spectrum, to `<name>.npz`.

    Skips the run entirely -- no eigensolves at all -- if the file already exists; pass
    `force=True` to redo it (e.g. after changing the scenario's start shape or `k`).
    """
    path = data_path(name)
    if os.path.exists(path) and not force:
        print(f'{name}: {path} exists, skipping the run (force=True to redo it)')
        return path

    fam = VertexPolygon(n, perim=PERIM)
    ev = SpectrumEvaluator(fam, n_eigs=n_eigs, warn_gap=0.0, degenerate_tol=0.0)
    p0 = param_vector(start, PERIM)
    p_target = param_vector(target, PERIM)
    rng = np.random.default_rng(seed)

    verts, lams, gap, delta, rho, mode, mm, jj = [], [], [], [], [], [], [], []

    def record(p, cur_lams, d, r, md, m_, j_):
        verts.append(fam.vertices(p))
        lams.append(np.asarray(cur_lams, dtype=float).copy())
        gap.append(_gap(cur_lams))
        delta.append(float(d))
        rho.append(float(r))
        mode.append(md)
        mm.append(int(m_))
        jj.append(int(j_))

    print(f'{name}: running...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lams0, _ = ev.eigenvalues(p0)
        record(p0, lams0, delta0, np.nan, 'init', 1, 0)

        def cb(p, F, model, d, r):
            record(p, ev._last[1], d, r, model['mode'], model.get('m', 1), model.get('j', 0))

        res = maximize_ratio(ev, p0, k=k, delta0=delta0, max_iter=max_iter, callback=cb, rng=rng)

        ev_target = SpectrumEvaluator(fam, n_eigs=n_eigs, warn_gap=0.0, degenerate_tol=0.0)
        exact_lams, _ = ev_target.eigenvalues(p_target)

    print(f'  {res.message}')
    print(f'  {res.nit} iterations, {len(verts)} frames, lam_{k+1}/lam_1 = {res.fun:.10f} '
         f'(exact {exact_ratio:.10f}, err {res.fun - exact_ratio:+.2e})')
    print(f'  {res.n_cluster_steps} cluster steps, {res.n_interior_steps} interior')

    os.makedirs(DATA_DIR, exist_ok=True)
    target_v = fam.vertices(p_target)
    np.savez(path,
            vertices=np.array(verts), lams=np.array(lams), gap=np.array(gap),
            delta=np.array(delta), rho=np.array(rho), mode=np.array(mode),
            m=np.array(mm), j=np.array(jj), target=target_v, exact_lams=np.asarray(exact_lams),
            exact_ratio=float(exact_ratio), k=int(k),
            fun=float(res.fun), nit=int(res.nit), message=str(res.message),
            n_cluster_steps=int(res.n_cluster_steps), n_interior_steps=int(res.n_interior_steps))
    print(f'  wrote {path}')
    return path


def _closed(v):
    return np.append(v, v[0])


def render(name, title, fps=10, hold=15, figsize=(8, 8), dpi=130):
    """Build the mp4 from `<name>.npz` alone -- no eigensolve, no `maximize_ratio` call."""
    d = np.load(data_path(name), allow_pickle=True)
    verts, lams, gap = d['vertices'], d['lams'], d['gap']
    mode, target = d['mode'], d['target']
    exact_ratio, exact_lams, k = float(d['exact_ratio']), d['exact_lams'], int(d['k'])
    n_it = len(verts)
    is_cluster = mode == 'cluster'

    fig, ((ax_s, ax_e), (ax_r, ax_g)) = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    # -- shape: current iterate against the exact maximizer --
    allv = np.concatenate([verts.reshape(-1), target])
    pad = 0.08*max(np.ptp(allv.real), np.ptp(allv.imag))
    ax_s.set_xlim(allv.real.min() - pad, allv.real.max() + pad)
    ax_s.set_ylim(allv.imag.min() - pad, allv.imag.max() + pad)
    ax_s.set_aspect('equal')
    ax_s.set_xticks([]); ax_s.set_yticks([])
    tv = _closed(target)
    ax_s.plot(tv.real, tv.imag, '--', color='0.6', lw=1.5, label='exact maximizer')
    (line_s,) = ax_s.plot([], [], '-', color='C0', lw=2)
    ax_s.legend(loc='upper right', fontsize=7, frameon=False)

    # -- eigenvalue trajectories, against the exact values at the target --
    ax_e.set_xlim(-0.5, n_it - 0.5)
    lo, hi = lams.min(), lams.max()
    ax_e.set_ylim(lo - 0.06*(hi - lo), hi + 0.10*(hi - lo))
    ax_e.set_title('eigenvalues', fontsize=9)
    ax_e.tick_params(labelsize=7)
    for j in range(lams.shape[1]):
        ax_e.axhline(float(exact_lams[j]), color='k', ls='--', lw=0.8, alpha=0.5)
    lines_e = [ax_e.plot([], [], lw=1.3, marker='.', ms=3, color=f'C{j}',
                         label=rf'$\lambda_{{{j+1}}}$')[0] for j in range(lams.shape[1])]
    ax_e.legend(loc='center right', fontsize=6, frameon=False)

    # -- |ratio - exact|, log scale --
    ratio = lams[:, k]/lams[:, 0]
    err = np.abs(ratio - exact_ratio)
    floor = max(err[err > 0].min() if np.any(err > 0) else 1e-16, 1e-16)
    ax_r.set_xlim(-0.5, n_it - 0.5)
    ax_r.set_ylim(0.3*floor, 3*max(err.max(), floor))
    ax_r.set_yscale('log')
    ax_r.set_xlabel('iteration', fontsize=8)
    ax_r.set_title(rf'$|\lambda_{{{k+1}}}/\lambda_1 - $ exact$|$', fontsize=9)
    ax_r.tick_params(labelsize=7)
    (line_r,) = ax_r.plot([], [], lw=1.3, marker='.', ms=3, color='C3')

    # -- relative spectral gap, colored by which local model was in force --
    gfloor = max(gap[np.isfinite(gap) & (gap > 0)].min(initial=1e-16), 1e-16)
    ghi = gap[np.isfinite(gap)].max() if np.any(np.isfinite(gap)) else 1.0
    ax_g.set_xlim(-0.5, n_it - 0.5)
    ax_g.set_ylim(0.3*gfloor, 3*ghi)
    ax_g.set_yscale('log')
    ax_g.set_xlabel('iteration', fontsize=8)
    ax_g.set_title('spectral gap (blue = smooth, orange = cluster)', fontsize=8)
    ax_g.tick_params(labelsize=7)
    (line_g_bg,) = ax_g.plot([], [], lw=0.8, color='0.75', zorder=1)
    (line_g_smooth,) = ax_g.plot([], [], '.', ms=4, color='C0', zorder=2)
    (line_g_cluster,) = ax_g.plot([], [], '.', ms=4, color='C1', zorder=2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(f):
        i = min(f, n_it - 1)
        v = _closed(verts[i])
        line_s.set_data(v.real, v.imag)
        ax_s.set_title(f'iteration {i}', fontsize=10)

        xs = np.arange(i + 1)
        for j, ln in enumerate(lines_e):
            ln.set_data(xs, lams[:i + 1, j])
        line_r.set_data(xs, np.maximum(err[:i + 1], 0.3*floor))

        g = np.maximum(gap[:i + 1], 0.3*gfloor)
        line_g_bg.set_data(xs, g)
        gs, gc = g.copy(), g.copy()
        gs[is_cluster[:i + 1]] = np.nan
        gc[~is_cluster[:i + 1]] = np.nan
        line_g_smooth.set_data(xs, gs)
        line_g_cluster.set_data(xs, gc)
        return [line_s, line_r, line_g_bg, line_g_smooth, line_g_cluster, *lines_e]

    anim = FuncAnimation(fig, update, frames=n_it + hold, blit=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{name}.mp4')
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f'  wrote {path}  ({n_it + hold} frames)')
    return path
