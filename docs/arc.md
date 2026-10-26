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

## The triangle grid (`benchmarks/specs/triangle_grid_v0.toml`)

2462 triangles (apex lattice h = 0.01, every angle >= 10 deg), 15 eigenvalues each, 200 array
tasks of ~13 points. Write the sbatch (never submits), look at it, submit:

```bash
uv run python -m benchmarks.family_grid slurm benchmarks/specs/triangle_grid_v0.toml --account <account> --partition <partition>
```

```bash
sbatch benchmarks/run/family_grid/triangle_grid_v0/triangle_grid_v0.sbatch
```

Watch, then pool and plot once tasks finish (both are safe to run on partial results):

```bash
squeue -u $USER
```

```bash
uv run python -m benchmarks.family_grid collect benchmarks/specs/triangle_grid_v0.toml
```

```bash
uv run python -m benchmarks.family_grid plot benchmarks/specs/triangle_grid_v0.toml
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
- Cost is set by the smallest angle and `n_eigs`, not the scale: ~5 s for triangles with
  angles pi/n, ~60 s generic, minutes below ~10 deg (measured on an M-series Mac, 2026-10-26).
  The angle floor is a spec key; lowering it is a new spec, not an edit.
- Rows record `spec_sha`, `douse_sha`, `chladni_sha` and `prec`; `collect` refuses rows from
  another `spec_sha`.
