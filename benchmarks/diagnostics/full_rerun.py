"""Re-run every row the guards excluded, on the fixed solver.

RECIPE FROM THE HANDOFF, and the backup is the point of it: the removed rows are the "before"
half of the comparison and a row whose value MOVES is the most interesting outcome available.
They go to `pre_rerun_<date>/` before anything is deleted, and `ratio_multistart` then resumes by
start index and refills exactly the gaps.

Cells are run in increasing cost order -- N=3 first at ~30 s per start, N=6 last at up to an
hour -- so that if this is interrupted the cheap evidence is already banked.
"""
import json, os, shutil, subprocess, sys, time
from collections import defaultdict
sys.path.insert(0, '/Users/hringer/projects/python/douse')
from douse import spectral_guard

LED = '/Users/hringer/projects/python/douse/benchmarks/run/ratio_multistart'
BACKUP = os.path.join(LED, 'pre_rerun_2026-08-27')


def flagged(row):
    if not row.get('ok', True):
        return 'crashed: ' + row['error'].split(':')[0]
    if row['worst_sigma'] > 1e-6:
        return f"drifted sigma {row['worst_sigma']:.1e}"
    if row.get('worst_weyl_deficit', 0) > spectral_guard.DEFICIT_TOL:
        return f"missed mode {row.get('worst_weyl_deficit'):+.2f}"
    return None


def main():
    os.makedirs(BACKUP, exist_ok=True)
    plan = {}
    for name in sorted(os.listdir(LED)):
        if not (name.startswith('N') and name.endswith('.jsonl')):
            continue
        path = os.path.join(LED, name)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        keep = [r for r in rows if not flagged(r)]
        drop = [r for r in rows if flagged(r)]
        if not drop:
            continue
        shutil.copy(path, os.path.join(BACKUP, name))
        with open(path, 'w') as fh:
            for r in keep:
                fh.write(json.dumps(r) + '\n')
        cell = name[:-6]
        N, k = int(cell[1:].split('_k')[0]), int(cell.split('_k')[1])
        plan[(N, k)] = [(r['start'], flagged(r), r.get('fun')) for r in drop]
        print(f"  {cell}: removed {len(drop)} row(s), {len(keep)} kept -> backup", flush=True)

    with open(os.path.join(BACKUP, 'removed.json'), 'w') as fh:
        json.dump({f'N{N}_k{k}': v for (N, k), v in plan.items()}, fh, indent=1)

    print(f"\n{len(plan)} cell(s) to refill; cheapest first\n", flush=True)
    for (N, k) in sorted(plan, key=lambda t: (t[0], t[1])):
        t0 = time.perf_counter()
        print(f"=== N={N} k={k} ({len(plan[(N, k)])} start(s) missing) {time.strftime('%H:%M')}",
              flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'benchmarks.ratio_multistart',
                        '--N', str(N), '--k', str(k), '--starts', '8',
                        '--max-iter', '400', '--workers', '5'],
                       cwd='/Users/hringer/projects/python/douse')
        print(f"=== N={N} k={k} done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
