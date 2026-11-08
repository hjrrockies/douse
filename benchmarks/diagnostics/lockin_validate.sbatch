#!/bin/bash
# The lock-in gate's validation: the 20 rows of lockin_rows.json on the gate's douse and chladni.
# Submit from the douse checkout. Writes to $DOUSE_RUNS/validation/lockin_v2.
#SBATCH --job-name=douse_lockin_validate
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=lockin_validate_%A_%a.out
cd "${DOUSE_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m benchmarks.diagnostics.lockin_validate run "$SLURM_ARRAY_TASK_ID" \
    --tau-g 1e-6 --out "${DOUSE_RUNS:?set DOUSE_RUNS}/validation/lockin_v2"
