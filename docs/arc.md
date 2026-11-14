# Running douse on VT ARC (Slurm)

The install is the CI recipe (`.github/workflows/tests.yml`): douse and chladni checked out SIDE
BY SIDE, because chladni is a path dependency (`../chladni`). lappy is no longer a dependency of
any kind (2026-11-13). A release at an OLDER douse commit (before that change, e.g.
`thesis-lock-1` at 2b66d9b) still lists `../lappy` as a path source, so for those, and only those,
a lappy checkout has to sit beside it for `uv sync` to resolve; lappy is never installed.

Placeholders: `<pid>` your VT PID, `<account>` your ARC allocation, `<cluster>` the login host
(e.g. `tinkercliffs1.arc.vt.edu` or `owl1.arc.vt.edu`), `<partition>` (e.g. `normal_q`). Check
them against ARC's docs or `sacctmgr show assoc user=$USER` once logged in.

## Where things live

```
~/projects/thesis/{douse,chladni,lappy}   clones to fetch into (lappy: old releases only); no job runs from these
~/projects/thesis/releases/<name>/        douse and chladni at exact commits, with their own .venv; jobs run here
~/projects/thesis/runs/                   $DOUSE_RUNS: ledgers and logs, same subfolders as the workstation's
```

chladni is an EDITABLE path dependency, so a `git pull` in its clone changes the code any
task that has not started yet will import. A release is checked out once and never pulled. Code
reaches the cluster only through GitHub, results leave only by rsync to the workstation, and
nothing is edited here. Results stay in home (no purge); `/scratch` deletes files at 90 days.

## Once per cluster

```bash
ssh <pid>@<cluster>
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
mkdir -p ~/projects/thesis && cd ~/projects/thesis
git clone https://github.com/hjrrockies/douse.git
git clone https://github.com/hjrrockies/chladni.git
git clone https://github.com/hjrrockies/lappy.git     # only for releases older than 2026-11-13
```

```bash
mkdir -p ~/projects/thesis/runs && echo 'export DOUSE_RUNS="$HOME/projects/thesis/runs"' >> ~/.bashrc && source ~/.bashrc
```

`.python-version` pins Python 3.12 (CI tests 3.11-3.13), and uv downloads it if the cluster
lacks it. Without the pin uv takes the newest Python it finds, and on a release newer than the
dependencies' wheels (3.15 on TinkerCliffs, 2026-10-30) it compiles cvxpy's `sparsediffpy`
from source and fails on BLAS.

To be mailed when a job ends or a task fails (one mail per array, not per task), set the
address once on the cluster; both job generators read it (`--mail` overrides). It is kept out of
the specs because the repository is public:

```bash
echo 'export DOUSE_SLURM_MAIL=<pid>@vt.edu' >> ~/.bashrc && source ~/.bashrc
```

## A release: one per set of runs

`<name>` is the tag the commits are to carry (`thesis-lock-1`), and `<douse commit>`,
`<chladni commit>` are full commits on GitHub. `uv.lock` is tracked, so `--frozen` installs what
was validated:

```bash
cd ~/projects/thesis && for r in douse chladni lappy; do git -C $r fetch --tags origin; done
```

```bash
R=releases/<name>; git -C douse worktree add --detach ../$R/douse <douse commit>; git -C chladni worktree add --detach ../$R/chladni <chladni commit>
```

For a release at a douse commit older than 2026-11-13 (whose `pyproject.toml` still has a
`lappy` path source), also: `git -C lappy worktree add --detach ../$R/lappy f3a0824`.

```bash
cd ~/projects/thesis/releases/<name>/douse && uv sync --frozen
```

Then the environment check, on a COMPUTE node (it solves the unit square and checks the LAPACK
route gsvd4py found; a failed solve here means nothing downstream is worth submitting):

```bash
srun -A <account> -p <partition> -t 0:15:00 --mem=2G .venv/bin/python -m benchmarks.env_check
```

Every command below runs from that `douse` directory. After a job, keep its accounting with its
results: `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS > "$DOUSE_RUNS/.../sacct.txt"`.

## The triangle grid: pilot first

`triangle_pilot_v1.toml` is the same recipe on 107 triangles (h = 0.05), one per task, ~20 min
wall. Run it first: it checks the install on real nodes and maps where chladni falls short of
`prec` (see Accuracy below) for ~1/25 of the full grid's cost. Every command below takes either
spec; swap `triangle_pilot_v1` for `triangle_grid_v1` (2462 triangles, h = 0.01, 200 tasks of
~13 points) once the pilot's `collect` looks right.

Write the sbatch (never submits), look at it, submit:

```bash
uv run python -m benchmarks.family_grid slurm benchmarks/specs/triangle_pilot_v1.toml --account <account> --partition <partition> --runner .venv/bin/python
```

```bash
sbatch "$DOUSE_RUNS/family_grid/triangle_pilot_v1/triangle_pilot_v1.sbatch"
```

Watch, then pool and plot once tasks finish (both are safe to run on partial results):

```bash
squeue -u $USER
```

```bash
uv run python -m benchmarks.family_grid collect benchmarks/specs/triangle_pilot_v1.toml
```

```bash
uv run python -m benchmarks.family_grid plot benchmarks/specs/triangle_pilot_v1.toml
```

A task that died (time limit, node failure) is resubmitted by index and resumes its own ledger:
`sbatch --array=17 <the .sbatch>`. `collect` lists how many points are missing.

When every task has finished, `collect` also counts SUSPECT rows: chladni's count audit fired,
or two adjacent simple levels sit within their own error bounds (one eigenvalue reported twice,
which shifts every level above it; one row in the 107-point triangle pilot). Re-solve just those,
with five more eigenvalues each, as a small array job, then `collect` again; plots and arrays use
the repaired rows and leave out any row still suspect:

```bash
uv run python -m benchmarks.family_grid slurm benchmarks/specs/triangle_pilot_v1.toml --repair 4 --runner .venv/bin/python --mem 1G --time 2:00:00
```

```bash
sbatch "$DOUSE_RUNS/family_grid/triangle_pilot_v1/repair/triangle_pilot_v1_repair.sbatch"
```

Bring results home (ledgers are JSONL, a few MB) into the workstation's one results directory,
`$DOUSE_RUNS` (`~/projects/thesis-runs`, outside every checkout; see its README). Every driver
reads and writes under `$DOUSE_RUNS` on both machines. Generate sbatch scripts ON the cluster,
since the default `--out` is an absolute path. One way, and never with `--delete`:

```bash
rsync -av <pid>@<cluster>:projects/thesis/runs/ ~/projects/thesis-runs/
```

## The triangle posterior (after the grid)

`benchmarks/triangle_posterior.py` does no solves: it interpolates the collected
`triangle_grid_v1` ledgers, so it needs them in `$DOUSE_RUNS/family_grid/triangle_grid_v1/`. The whole of `triangle_posterior_v1` is
under 1 CPU-h (5454 posteriors, ~0.6 s each); 20 tasks finish in a few minutes. Build the
surrogate once on the login node (~1 minute), then write the sbatch and submit:

```bash
uv run python -m benchmarks.triangle_posterior prepare benchmarks/specs/triangle_posterior_v1.toml
```

```bash
uv run python -m benchmarks.triangle_posterior slurm benchmarks/specs/triangle_posterior_v1.toml --runner .venv/bin/python
```

```bash
sbatch "$DOUSE_RUNS/triangle_posterior/triangle_posterior_v1/triangle_posterior_v1.sbatch"
```

```bash
uv run python -m benchmarks.triangle_posterior plot benchmarks/specs/triangle_posterior_v1.toml
```

`plot` prints the coverage table and writes `figures/` (posterior maps, coverage, Kendall
measure of the 90% region, the interpolant's cross-validated error) beside the ledgers.

## Notes

- `--runner .venv/bin/python` (the venv `uv sync` built) rather than `uv run`, always: a hundred
  tasks starting at once would each re-check the lock on the shared filesystem.
- One CPU per task, BLAS threads pinned to 1: each point is one serial solve; a BLAS sizing
  itself from the node oversubscribes it.
- Cost, M-series Mac, 2026-10-26, eight solves at once: ~5 s for triangles with angles pi/n,
  15 s-6 min for generic ones (it is NOT a simple function of the smallest angle), 6-15 min at
  height 0.10-0.15. Rough total for triangle_grid_v1: 70-150 CPU-h; the pilot's `collect`
  prints the measured figure. Peak memory per task is not measured yet; `seff <jobid>` after
  the pilot says, and the full grid's `slurm_mem` should be set from it.
- ACCURACY. `bound` in a row is chladni's ABSOLUTE Moler-Payne bound; `collect` reports it
  relative. Measured locally: most triangles meet 1e-10 on all 15 levels (relative ~1e-10);
  flat ones miss the margin rule at ~3e-9 relative; and at least one, apex (0.1, 0.25), came
  back from the planner's first rung with 0 of 15 met and relative bounds 1e-3 to 5e-2. The
  Weyl deficit does not flag these (no mode is missing, the values are loose). Filter on
  `met`/`bound` before any figure or lookup; the pilot says how common the bad case is.
  The angle floor is a spec key; lowering it is a new spec, not an edit.
- Rows record `spec_sha`, `douse_sha`, `chladni_sha` and `prec`; `collect` refuses rows from
  another `spec_sha`.
