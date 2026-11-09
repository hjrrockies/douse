#!/bin/bash
# The 20 rows of `lockin_validate`, one per array task. Run from the douse checkout, on `thesis`,
# with chladni at the commit being validated:
#     mkdir -p "$DOUSE_RUNS/lockin_validate/logs"
#     sbatch --account=<account> --partition=<partition> \
#         --output="$DOUSE_RUNS/lockin_validate/logs/%A_%a.out" benchmarks/diagnostics/lockin_validate.sbatch
# then, once they finish:
#     uv run python -m benchmarks.diagnostics.lockin_validate compare "$DOUSE_RUNS/lockin_validate"
#SBATCH --job-name=douse_lockin_validate
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -m benchmarks.diagnostics.lockin_validate run "$SLURM_ARRAY_TASK_ID" --out "$DOUSE_RUNS/lockin_validate"
