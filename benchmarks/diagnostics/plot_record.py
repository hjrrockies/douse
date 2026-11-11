"""Draw, or summarize, one run's record (`douse.record`), with no re-solves.

    python -m benchmarks.diagnostics.plot_record RECORD.jsonl [--out STEM]   figure (.png, .pdf)
    python -m benchmarks.diagnostics.plot_record RECORD.jsonl --summary      text, for a live run

The figure, one column per iteration of the loop:
  shapes        the start, the accepted iterates (faint) and the last one
  F             the objective at the iterate each iteration ended on
  levels        every computed level over lam_1, at the accepted iterates; the active cluster marked
  radius        delta, coloured by the model's m (the window the radius sees) and by resolved m
  rho           actual over predicted gain, clipped
  criticality   crit against eps_f, sqrt(2 eps_f) and eps_g: where the run stands against what
                its solves can resolve
  geometry      the smallest and largest interior angle, the shortest edge
  cost          seconds per evaluation, by the kind of solve
"""
import argparse
import pathlib
import sys

import numpy as np

from douse.record import read

KIND_COLOURS = dict(cold='tab:blue', warm='tab:green', plan='tab:orange', replan='tab:red')


def _key(p):
    return None if p is None else tuple(np.round(np.asarray(p, dtype=float), 15))


def summary(r, last=10):
    head = r['head'][0] if r['head'] else {}
    out = [f"{head.get('N')}-gon, lam_{(head.get('k') or 0) + 1}/lam_1, start {head.get('start')}"
           f"  douse {head.get('douse_sha')} chladni {head.get('chladni_sha')}"
           f"  ({len(r['head'])} head line(s): {len(r['head']) - 1} resume(s))",
           f"{len(r['iter'])} iterations, {len(r['eval'])} evaluations, {len(r['plan'])} plan(s), "
           f"{sum(len(e['solves']) for e in r['eval'])} solves"
           + (f", {len(r['error'])} RECORDER ERRORS" if r.get('error') else '')]
    if r['iter']:
        out.append(f"{'it':>5} {'F':>15} {'delta':>9} {'rho':>7} {'mode':>7} {'m':>2} {'res':>3} "
                   f"{'crit':>9} {'eps_f':>9} {'eps_g':>9} {'min/max angle':>14} {'t':>8}")
        for i in r['iter'][-last:]:
            m = i['model'] or {}
            res = m.get('resolved')
            g = i.get('geometry') or {}
            rho = i['rho'] if i['rho'] is not None and i['rho'] > -1e300 else float('nan')
            out.append(f"{i['it']:5d} {i['F']:15.10f} {i['delta']:9.2e} {rho:+7.2f} "
                       f"{m.get('mode', '-'):>7} {m.get('m', '-'):>2} "
                       f"{'-' if not res else res[1] - res[0]:>3} {i['crit']:9.2e} "
                       f"{_f(i['eps_f']):>9} {_f(i['eps_g']):>9} "
                       f"{g.get('min_angle', np.nan):6.1f}/{g.get('max_angle', np.nan):6.1f} "
                       f"{i['t']:8.0f}")
    if r['result']:
        res = r['result'][-1]
        out.append(f"ended: F = {res.get('fun')}  {res.get('convergence')}  "
                   f"{res.get('floor_reason')}  {str(res.get('message'))[:140]}")
    return '\n'.join(out)


def _f(x):
    return '-' if x is None or not np.isfinite(x) else f'{x:.2e}'


def figure(r, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    head = r['head'][0] if r['head'] else {}
    its = r['iter']
    it = np.array([i['it'] for i in its])
    evals = {_key(e['p']): e for e in r['eval'] if e.get('lam') is not None}

    fig, ax = plt.subplots(2, 4, figsize=(21, 9.5))
    fig.suptitle(f"{head.get('N')}-gon, max lam_{(head.get('k') or 0) + 1}/lam_1, start "
                 f"{head.get('start')}, chart {head.get('chart')}  --  {len(its)} iterations, "
                 f"{len(r['eval'])} evaluations  (douse {head.get('douse_sha')}, chladni "
                 f"{head.get('chladni_sha')}; from the record, no re-solves)")

    a = ax[0, 0]
    shapes = [np.asarray(e['v'][0]) + 1j*np.asarray(e['v'][1]) for e in r['eval']
              if e.get('v') is not None and e.get('lam') is not None][:1]
    acc = [i['geometry']['v'] for i in its if i['moved'] and (i.get('geometry') or {}).get('v')]
    acc = [np.asarray(v[0]) + 1j*np.asarray(v[1]) for v in acc]
    for v in acc[::max(1, len(acc)//25)]:
        c = np.r_[v, v[:1]]
        a.plot(c.real, c.imag, '-', color='0.75', lw=0.5)
    for v, sty, lab in [(shapes[0] if shapes else None, '--', 'start'),
                        (acc[-1] if acc else None, '-', 'last iterate')]:
        if v is not None:
            c = np.r_[v, v[:1]]
            a.plot(c.real, c.imag, sty, lw=1.6, label=lab)
            a.plot(v[0].real, v[0].imag, 'o', ms=4, color=a.lines[-1].get_color())
    a.set_aspect('equal'); a.legend(fontsize=8); a.set_title('shapes (grey: accepted iterates)')

    a = ax[0, 1]
    a.plot(it, [i['F'] for i in its], '.-', ms=3, lw=0.8)
    a.set_xlabel('iteration'); a.set_title('F at the iterate')

    a = ax[0, 2]
    rows = []
    for i in its:
        e = evals.get(_key(i['p']))
        if e is None:
            continue
        lam = np.repeat(np.asarray(e['lam']), np.asarray(e['mult']))
        rows.append((i['it'], lam/lam[0]))
    for j in range(max((len(x) for _, x in rows), default=0)):
        pts = [(t, x[j]) for t, x in rows if j < len(x)]
        a.plot(*zip(*pts), '.', ms=2, label=f'lam_{j + 1}' if j < 12 else None)
    for i in its:
        if i['model']:
            lo, hi = i['model']['cluster']
            e = evals.get(_key(i['p_model']))
            if e is not None:
                lam = np.repeat(np.asarray(e['lam']), np.asarray(e['mult']))
                a.plot([i['it']]*2, [lam[lo]/lam[0], lam[hi - 1]/lam[0]], '-', color='k', lw=2,
                       alpha=0.25)
    a.set_xlabel('iteration'); a.set_title('levels / lam_1 (bar: the model\'s cluster)')
    a.legend(fontsize=6, ncol=2)

    a = ax[0, 3]
    ms = np.array([i['model']['m'] if i['model'] else 0 for i in its])
    rm_ = np.array([(i['model']['resolved'][1] - i['model']['resolved'][0]) if i['model'] else 0
                    for i in its])
    dl = np.array([i['delta'] for i in its])
    a.semilogy(it, dl, '-', color='0.7', lw=0.6)
    sc = a.scatter(it, dl, c=ms, s=12, cmap='viridis', vmin=1, vmax=max(2, ms.max()))
    fig.colorbar(sc, ax=a, label="model's m", pad=0.15)
    b = a.twinx(); b.step(it, rm_, where='mid', color='tab:red', lw=0.8); b.set_ylabel('resolved m', color='tab:red')
    a.set_xlabel('iteration'); a.set_title('trust radius')

    a = ax[1, 0]
    rho = np.array([i['rho'] if i['rho'] is not None and i['rho'] > -1e300 else np.nan for i in its])
    a.plot(it, np.clip(rho, -5, 2), '.', ms=4)
    a.axhline(0, color='k', lw=0.5); a.axhline(0.1, color='k', lw=0.5, ls=':')
    a.set_ylim(-5.2, 2.2); a.set_xlabel('iteration'); a.set_title('rho (clipped; dotted: eta)')

    a = ax[1, 1]
    for key, sty in [('crit', '.'), ('eps_g', '-'), ('eps_f', '-')]:
        y = np.array([i[key] if i[key] is not None else np.nan for i in its], dtype=float)
        a.semilogy(it, np.where(y > 0, y, np.nan), sty, ms=3, lw=0.8, label=key)
    ef = np.array([i['eps_f'] if i['eps_f'] is not None else np.nan for i in its], dtype=float)
    a.semilogy(it, np.sqrt(2*ef), '--', lw=0.8, label='sqrt(2 eps_f)')
    a.set_xlabel('iteration'); a.legend(fontsize=7); a.set_title('criticality and error floors')

    a = ax[1, 2]
    g = [i.get('geometry') or {} for i in its]
    a.plot(it, [x.get('min_angle', np.nan) for x in g], '-', label='min angle')
    a.plot(it, [x.get('max_angle', np.nan) for x in g], '-', label='max angle')
    a.set_ylabel('degrees'); a.legend(fontsize=7, loc='upper left')
    b = a.twinx(); b.plot(it, [x.get('min_edge', np.nan) for x in g], '-', color='tab:green')
    b.set_ylabel('shortest edge', color='tab:green')
    a.set_xlabel('iteration'); a.set_title('geometry at the iterate')

    a = ax[1, 3]
    for n, e in enumerate(r['eval'], 1):
        for s in e['solves']:
            a.semilogy(n, max(s['seconds'], 1e-2), 'o', ms=3,
                       color=KIND_COLOURS.get(s['kind'], 'k'))
    for kind, c in KIND_COLOURS.items():
        a.plot([], [], 'o', color=c, label=kind)
    a.legend(fontsize=7); a.set_xlabel('evaluation'); a.set_ylabel('seconds')
    a.set_title('cost per solve')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=120)
    return f'{stem}.png'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('record')
    ap.add_argument('--out', default=None, help='figure stem (default: beside the record)')
    ap.add_argument('--summary', action='store_true', help='print the last iterations instead')
    ap.add_argument('--last', type=int, default=10)
    a = ap.parse_args(argv)
    r = read(a.record)
    if a.summary:
        print(summary(r, a.last))
        return
    stem = a.out or str(pathlib.Path(a.record).with_suffix(''))
    print(figure(r, stem))


if __name__ == '__main__':
    sys.exit(main())
