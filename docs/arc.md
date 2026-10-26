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
mkdir -p ~/thesis && cd ~/thesis
git clone https://github.com/hjrrockies/douse.git
git clone -b fix/douse-pipeline-2026-10-14 https://github.com/hjrrockies/chladni.git
git clone https://github.com/hjrrockies/lappy.git
```

```bash
cd ~/thesis/douse && git checkout <the branch with benchmarks/family_grid.py> && uv sync
```

Then the environment check, on a COMPUTE node (it solves the unit square and checks the LAPACK
route gsvd4py found; a failed solve here means nothing downstream is worth submitting):

```bash
srun -A <account> -p <partition> -t 0:15:00 --mem=2G uv run python -m benchmarks.env_check
```

## The triangle grid: pilot first

`triangle_pilot_v0.toml` is the same recipe on 107 triangles (h = 0.05), one per task, ~20 min
wall. Run it first: it checks the install on real nodes and maps where chladni falls short of
`prec` (see Accuracy below) for ~1/25 of the full grid's cost. Every command below takes either
spec; swap `triangle_pilot_v0` for `triangle_grid_v0` (2462 triangles, h = 0.01, 200 tasks of
~13 points) once the pilot's `collect` looks right.

Write the sbatch (never submits), look at it, submit:

```bash
uv run python -m benchmarks.family_grid slurm benchmarks/specs/triangle_pilot_v0.toml --account <account> --partition <partition>
```

```bash
sbatch benchmarks/run/family_grid/triangle_pilot_v0/triangle_pilot_v0.sbatch
```

Watch, then pool and plot once tasks finish (both are safe to run on partial results):

```bash
squeue -u $USER
```

```bash
uv run python -m benchmarks.family_grid collect benchmarks/specs/triangle_pilot_v0.toml
```

```bash
uv run python -m benchmarks.family_grid plot benchmarks/specs/triangle_pilot_v0.toml
```

A task that died (time limit, node failure) is resubmitted by index and resumes its own ledger:
`sbatch --array=17 <the .sbatch>`. `collect` lists how many points are missing.

Bring results home (ledgers are JSONL, a few MB):

```bash
rsync -av <pid>@<cluster>:thesis/douse/benchmarks/run/family_grid/ benchmarks/run/family_grid/
```

## Notes

- One CPU per task, BLAS threads pinned to 1: each point is one serial solve; a BLAS sizing
  itself from the node oversubscribes it.
- Cost, M-series Mac, 2026-10-26, eight solves at once: ~5 s for triangles with angles pi/n,
  15 s-6 min for generic ones (it is NOT a simple function of the smallest angle), 6-15 min at
  height 0.10-0.15. Rough total for triangle_grid_v0: 70-150 CPU-h; the pilot's `collect`
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
