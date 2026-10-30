# Running douse on VT ARC (Slurm)

The install is the CI recipe (`.github/workflows/tests.yml`): douse, chladni and lappy checked
out SIDE BY SIDE, because chladni and lappy are path dependencies (`../chladni`, `../lappy`).
lappy is never installed; its directory only has to exist for `uv sync` to resolve.

Placeholders: `<pid>` your VT PID, `<account>` your ARC allocation, `<cluster>` the login host
(e.g. `tinkercliffs1.arc.vt.edu` or `owl1.arc.vt.edu`), `<partition>` (e.g. `normal_q`). Check
them against ARC's docs or `sacctmgr show assoc user=$USER` once logged in.

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
git clone https://github.com/hjrrockies/lappy.git
```

```bash
cd ~/projects/thesis/douse && git checkout claude/intelligent-turing-9278ef && uv sync
```

`.python-version` pins Python 3.12 (CI tests 3.11-3.13), and uv downloads it if the cluster
lacks it. Without the pin uv takes the newest Python it finds, and on a release newer than the
dependencies' wheels (3.15 on TinkerCliffs, 2026-10-30) it compiles cvxpy's `sparsediffpy`
from source and fails on BLAS.

chladni is cloned at `main`, which has the opt-in `coarse_bracket` path (chladni PR #20) that
the v1 specs use. The douse branch is douse PR #2; once it merges, use douse `main`.

Then the environment check, on a COMPUTE node (it solves the unit square and checks the LAPACK
route gsvd4py found; a failed solve here means nothing downstream is worth submitting):

```bash
srun -A <account> -p <partition> -t 0:15:00 --mem=2G uv run python -m benchmarks.env_check
```

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
sbatch benchmarks/run/family_grid/triangle_pilot_v1/triangle_pilot_v1.sbatch
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
sbatch benchmarks/run/family_grid/triangle_pilot_v1/repair/triangle_pilot_v1_repair.sbatch
```

Bring results home (ledgers are JSONL, a few MB):

```bash
rsync -av <pid>@<cluster>:projects/thesis/douse/benchmarks/run/family_grid/ benchmarks/run/family_grid/
```

## Notes

- `--runner .venv/bin/python` (the venv `uv sync` built) rather than `uv run`: a hundred tasks
  starting at once would each re-check the lock on the shared filesystem.
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
