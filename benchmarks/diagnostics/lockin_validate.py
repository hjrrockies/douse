"""Does the lock-in gate's optimizer finish the runs that stalled? A side ledger, never a result.

Re-runs a fixed sample of `thesis_ratio_v2`'s stalled rows (`lockin_rows.json`, 20 rows: same N,
k, start and budget, ended "trust radius collapsed", on a refusal streak, or unverified, and
under about 8 hours then) with the finishing precision the thesis runs will use.

    python -m benchmarks.diagnostics.lockin_validate list
    python -m benchmarks.diagnostics.lockin_validate run 0 --tau-g 1e-6 --out DIR
    python -m benchmarks.diagnostics.lockin_validate compare DIR

`compare` applies the plan's pass criteria: no row ends collapsed, no row loses more than 1e-6
against its v2 value, refusal-streak endings drop.
"""
import argparse
import dataclasses
import functools
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROWS = HERE/'lockin_rows.json'
SPEC = HERE.parent/'specs'/'thesis_ratio_v2.toml'
PREC_FINAL = [1e-12, 1e-11, 1e-10]
LOSS_TOL = 1e-6


def rows():
    return json.loads(ROWS.read_text())


def run(index, tau_g, out):
    import benchmarks.ratio_multistart as rm
    from douse import spec as spec_mod
    from douse.cluster_opt import maximize_ratio
    r = rows()[index]
    spec = dataclasses.replace(spec_mod.load(SPEC), prec_final=PREC_FINAL)
    rm.maximize_ratio = functools.partial(maximize_ratio, tau_g=tau_g)
    t0 = time.perf_counter()
    row = rm.run_one(r['N'], r['k'], r['start'], spec.max_iter, spec.seed0, spec.delta0,
                     prec=spec.prec, backend=spec.backend, spec=spec)
    row.update(index=index, tau_g=tau_g, v2_fun=r['fun'], v2_kind=r['kind'],
               v2_seconds=r['seconds'], seconds_rerun=round(time.perf_counter() - t0, 1))
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out/f'row{index:02d}_tau{tau_g:.0e}.jsonl', 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    print(row['fun'], row['message'], flush=True)


def _ending(r):
    m = r.get('message', '')
    if 'error' in r and 'fun' not in r:
        return 'error'
    if not r.get('verified', True):
        return 'unverified'
    if 'collapsed' in m:
        return 'collapsed'
    if 'stationary to within' in m or 'accuracy floor' in m:
        return 'floor'
    if 'streak' in m or 'in a row' in m or 'refus' in m:
        return 'refused'
    if 'max_iter' in m:
        return 'max_iter'
    return 'certified'


def compare(folder):
    new = [json.loads(line) for f in sorted(pathlib.Path(folder).glob('row*.jsonl'))
           for line in open(f) if line.strip()]
    print(f"{'row':>3} {'tau':>6} {'N':>2} {'k':>2} {'st':>2} {'v2':>10} {'now':>10} "
          f"{'F change':>10} {'eps_g':>8} {'final':>6} {'h now':>6} {'h v2':>6}")
    losses, collapsed = [], []
    for r in new:
        end = _ending(r)
        d = r.get('fun', np.nan) - r['v2_fun']
        print(f"{r['index']:3d} {r['tau_g']:6.0e} {r['N']:2d} {r['k']:2d} {r['start']:2d} "
              f"{r['v2_kind']:>10} {end:>10} {d:+10.2e} {r.get('eps_g') or np.nan:8.1e} "
              f"{str(r.get('prec_final_reached')):>6} {r.get('seconds_rerun', 0)/3600:6.1f} "
              f"{r['v2_seconds']/3600:6.1f}")
        if not d >= -LOSS_TOL:
            losses.append((r['index'], r['tau_g'], d))
        if end in ('collapsed', 'error'):
            collapsed.append((r['index'], r['tau_g'], end))
    v2 = sum(r['kind'] == 'refused' for r in rows())
    now = sum(_ending(r) == 'refused' for r in new)
    print(f"\n{len(new)} rows. collapsed or errored: {collapsed or 'none'}. lost more than "
          f"{LOSS_TOL:g}: {losses or 'none'}. refusal endings: {now} (v2 sample: {v2}).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    p = sub.add_parser('run')
    p.add_argument('index', type=int)
    p.add_argument('--tau-g', type=float, default=1e-6)
    p.add_argument('--out', required=True)
    p = sub.add_parser('compare')
    p.add_argument('folder')
    args = ap.parse_args(argv)
    if args.cmd == 'list':
        for i, r in enumerate(rows()):
            print(i, r)
    elif args.cmd == 'run':
        run(args.index, args.tau_g, args.out)
    else:
        compare(args.folder)


if __name__ == '__main__':
    main()
