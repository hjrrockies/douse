# Run specifications

One TOML file per sweep (`douse/spec.py`): the cells, the start sampler's seed and count, the
optimizer's budget and radius, the solver's precision and backend, the constraint family. The
driver runs a spec with `python -m benchmarks.ratio_multistart --spec <file>`, and every row it
writes carries `spec` (the name) and `spec_sha` (a content hash), so `summarize_grid` can report
which recipes wrote a cell and a reader can refuse to pool different ones. The code's own
identity (`douse_sha`, `chladni_sha`) is recorded beside the spec, not inside it.

| spec | what |
|---|---|
| `thesis_ratio_v0.toml` | the cells the laptop can reach: N = 3..6, k = 2..4, 8 starts, 400 iterations, 1e-10 |
| `triangle_pilot_v0.toml`, `triangle_grid_v0.toml` | `family_grid`: lam_1..15 over triangles, h = 0.05 / 0.01, 1e-10 |
| `triangle_pilot_v1.toml`, `triangle_grid_v1.toml` | the same with chladni's `coarse_bracket` (sharp triangles certify); use these |

## On the cluster (Slurm)

`python -m benchmarks.slurm <spec> [--dry-run]` writes one array job per cell
(`benchmarks/run/slurm/<spec name>/<cell>.sbatch`): `--array=0-(starts-1)`, one start per task,
BLAS threads pinned to 1, each task appending to its own ledger `<cell>/task_<i>.jsonl`. It never
submits; it prints the `sbatch` lines. `summarize_grid --dir benchmarks/run/slurm/<spec name>`
pools the task ledgers per cell. Run `python -m benchmarks.env_check` on a new machine first.

A spec may name its per-task resources -- `slurm_mem` (`"4G"`), `slurm_time` (`"24:00:00"`),
`slurm_partition`, `slurm_account`, all strings as `sbatch` takes them. They say where a spec
runs, not what it computes, so they are NOT part of the content hash: adding them leaves
`spec_sha` unchanged. Without them the jobs get 4G and 24 hours; `--mem`/`--time` override.
