#!/bin/bash
# The 20 rows of `lockin_validate`, one per array task. Run from a release checkout
# (`releases/<name>/douse`, see docs/arc.md). `LOCKIN_RUN` names this attempt, so a rerun cannot
# append to an earlier one's rows:
#     export LOCKIN_RUN=<name>
#     mkdir -p "$DOUSE_RUNS/lockin_validate/$LOCKIN_RUN/logs"
#     sbatch --account=<account> --partition=<partition> \
#         --output="$DOUSE_RUNS/lockin_validate/$LOCKIN_RUN/logs/%A_%a.out" benchmarks/diagnostics/lockin_validate.sbatch
# then, once they finish and are rsynced to the workstation:
#     python -m benchmarks.diagnostics.lockin_validate compare "$DOUSE_RUNS/lockin_validate/$LOCKIN_RUN"
#SBATCH --job-name=douse_lockin_validate
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m benchmarks.diagnostics.lockin_validate run "$SLURM_ARRAY_TASK_ID" --out "$DOUSE_RUNS/lockin_validate/$LOCKIN_RUN" --floor-policy "${LOCKIN_FLOOR_POLICY:-value_limit}"
