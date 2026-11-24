"""Resubmit the unfinished tasks of a spec's Slurm runs, each at the memory it needs.  (2026-11-24)

    python -m benchmarks.resume $DOUSE_RUNS/specs/<spec>.toml [--dir DIR] [--qos tc_normal_long]

Prints one `sbatch --array=<tasks> --mem=<mem> <cell>.sbatch` line per cell and memory. IT NEVER
SUBMITS (like `benchmarks/slurm.py`): run the lines from the douse checkout the tasks should run
on -- the scripts `cd` to the directory `sbatch` is run from, so the release you stand in is the
code a resumed task gets. A resumed task picks up its own checkpoint (`ratio_multistart`).

WHICH TASKS: every start of every cell with no finished row in its task ledger
(`<dir>/<cell>/task_<i>.jsonl`, the rule `ratio_multistart` resumes by: an ok row of this spec).
A crashed row is retried like a missing one. Tasks still queued or running (`squeue`) are left
alone, so a resubmit cannot race a live task over one checkpoint; `--no-squeue` skips the check
(off the cluster, or in tests).

WHICH MEMORY: the cell's figure from the spec (`slurm_mem`, a table by N or one figure), doubled
for every out-of-memory kill in that task's logs (`<dir>/logs/<cell>_<job>_<i>.out`, Slurm's
"oom_kill event" line), up to `--cap`. A task that outgrew its tier comes back one tier up from
its checkpoint, at most ~30 minutes lost (`runs.Checkpoint` flushes every 1800 s, atomically, so
a kill mid-write leaves the previous state); the rest of the cell keeps the small figure. That is
the trade the tiers make: most tasks at the figure their N needs, the outliers paying a restart.
"""
import argparse
import collections
import getpass
import os
import pathlib
import re
import shutil
import subprocess

from douse import ledger, spec as spec_mod

from .slurm import DEFAULT_MEM, cell_name

OOM_MARK = 'oom_kill event'
DEFAULT_CAP = '16G'


def gib(mem):
    """`'512M'`, `'2G'`, `'1T'` (sbatch's units; a bare number is megabytes) in GiB."""
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)B?', str(mem).strip().upper())
    if not m:
        raise ValueError(f'not a memory size sbatch takes: {mem!r}')
    return float(m.group(1)) * {'K': 2**-20, 'M': 2**-10, '': 2**-10, 'G': 1, 'T': 2**10}[m.group(2)]


def as_mem(g):
    """GiB as sbatch takes it: whole gigabytes where possible, else megabytes."""
    return f'{int(g)}G' if g == int(g) else f'{int(round(g * 1024))}M'


def n_tasks(spec, N, k):
    """A seeded cell has one task per seed, as `slurm.script` writes its array."""
    return len(spec.seed_polygons(N, k)) if spec.seeds else spec.starts


def finished(spec, path):
    """Starts with an ok row of this spec in a task ledger (`ratio_multistart`'s resume rule)."""
    if not os.path.exists(path):
        return set()
    return {r['start'] for r in ledger.read_rows(path, dedupe=False)
            if r.get('ok', True) and r.get('start') is not None and r.get('spec_sha') == spec.sha}


def oom_kills(logs, cell, i):
    """How many of task `i`'s logs end in an out-of-memory kill."""
    n = 0
    for f in pathlib.Path(logs).glob(f'{cell}_*_{i}.out'):
        if re.fullmatch(rf'{re.escape(cell)}_\d+_{i}\.out', f.name):
            with open(f, errors='replace') as fh:
                n += any(OOM_MARK in line for line in fh)
    return n


def active(spec):
    """`{(cell, task)}` queued or running for this spec, from `squeue` (job names are
    `douse_<spec>_<cell>`, as `slurm.script` writes them)."""
    if not shutil.which('squeue'):
        raise SystemExit('squeue not found: run on the cluster, or pass --no-squeue')
    out = subprocess.run(['squeue', '-u', getpass.getuser(), '-h', '-r', '-o', '%j %K'], check=True,
                         capture_output=True, text=True).stdout
    prefix = f'douse_{spec.name}_'
    live = set()
    for line in out.splitlines():
        name, _, task = line.partition(' ')
        if name.startswith(prefix) and task.strip().isdigit():
            live.add((name[len(prefix):], int(task)))
    return live


def plan(spec, run_dir, live=frozenset(), cap=DEFAULT_CAP, mem=None):
    """`[(cell, mem, [tasks])]` to resubmit, in the spec's cell order, one entry per memory."""
    logs = os.path.join(run_dir, 'logs')
    out = []
    for N, k in spec.cells:
        cell = cell_name(spec, N, k)
        base = gib(mem or spec.mem_for(N) or DEFAULT_MEM)
        done = set()
        for f in pathlib.Path(run_dir, cell).glob('task_*.jsonl'):
            if re.fullmatch(r'task_\d+\.jsonl', f.name):
                done |= finished(spec, str(f))
        by_mem = collections.defaultdict(list)
        for i in range(n_tasks(spec, N, k)):
            if i in done or (cell, i) in live:
                continue
            g = min(base * 2 ** oom_kills(logs, cell, i), max(base, gib(cap)))
            by_mem[g].append(i)
        out += [(cell, as_mem(g), tasks) for g, tasks in sorted(by_mem.items())]
    return out


def array_spec(tasks):
    """`[0, 1, 2, 5]` as `0-2,5`."""
    runs, start = [], None
    for a, b in zip(tasks, tasks[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            runs.append(str(start) if start == a else f'{start}-{a}')
            start = None
    return ','.join(runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('spec', help='the spec the runs were made from')
    ap.add_argument('--dir', default=None, help='the runs (default $DOUSE_RUNS/slurm/<spec name>)')
    ap.add_argument('--mem', default=None, help="override the spec's memory as the base figure")
    ap.add_argument('--cap', default=DEFAULT_CAP, help=f'never ask for more (default {DEFAULT_CAP})')
    ap.add_argument('--qos', default=None, help='e.g. tc_normal_long for a run over 7 days')
    ap.add_argument('--no-squeue', action='store_true', help='do not skip queued/running tasks')
    args = ap.parse_args(argv)

    spec = spec_mod.load(args.spec)
    run_dir = args.dir or spec_mod.run_dir('slurm', spec.name)
    live = frozenset() if args.no_squeue else active(spec)
    lines = plan(spec, run_dir, live, args.cap, args.mem)
    qos = f' --qos={args.qos}' if args.qos else ''
    n = sum(len(t) for _, _, t in lines)
    print(f'# {spec.name} @ {spec.sha}: {n} unfinished task(s) in {len({c for c, _, _ in lines})} '
          f'cell(s){"" if args.no_squeue else f", {len(live)} queued or running left alone"}. '
          'Submit from the douse checkout the tasks should run on:')
    for cell, mem, tasks in lines:
        print(f'sbatch --array={array_spec(tasks)} --mem={mem}{qos} {run_dir}/{cell}.sbatch')


if __name__ == '__main__':
    main()
