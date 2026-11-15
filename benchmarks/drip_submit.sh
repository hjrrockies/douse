#!/usr/bin/env bash
# Submit a directory of sbatch scripts as ARC's per-user queue limit allows.
#
# normal_q's partition QOS (tc_normal_base) caps a user at 1000 queued-or-running jobs, array
# tasks counted one by one and jobs on tc_normal_long included. A spec with more tasks than the
# room left gets "QOSMaxSubmitJobPerUserLimit" on the overflow. This loop submits each script
# once there is room for its whole array, marks it in <dir>/submitted/, and sleeps between passes.
#
#   nohup benchmarks/drip_submit.sh $DOUSE_RUNS/slurm/<spec name> [extra sbatch args] \
#       > $DOUSE_RUNS/slurm/<spec name>/drip.log 2>&1 &
#
# Scripts whose job is already in the queue are marked on the first pass, never resubmitted. To
# resubmit a whole spec later (the week-2 resume), delete <dir>/submitted/ first.
set -u
dir=${1:?usage: drip_submit.sh <slurm dir> [sbatch args]}; shift
limit=${DRIP_LIMIT:-1000}
pause=${DRIP_PAUSE:-1800}
mkdir -p "$dir/submitted" "$dir/logs"

name_of() { sed -n 's/^#SBATCH --job-name=//p' "$1"; }
tasks_of() {                                  # array size, 1 for a plain job
    local a; a=$(sed -n 's/^#SBATCH --array=//p' "$1" | sed 's/%.*//')
    [ -z "$a" ] && { echo 1; return; }
    echo "$a" | tr ',' '\n' | awk -F- '{n += (NF == 2 ? $2 - $1 + 1 : 1)} END {print n}'
}

queued=$(squeue -u "$USER" -h -o %j | sort -u)
for f in "$dir"/*.sbatch; do
    grep -qxF "$(name_of "$f")" <<<"$queued" && touch "$dir/submitted/$(basename "$f")"
done

while :; do
    left=0
    for f in "$dir"/*.sbatch; do
        [ -e "$dir/submitted/$(basename "$f")" ] && continue
        left=$((left + 1))
        n=$(squeue -u "$USER" -h -r | wc -l)
        [ $((n + $(tasks_of "$f"))) -le "$limit" ] || break
        if sbatch "$@" "$f"; then
            touch "$dir/submitted/$(basename "$f")"; left=$((left - 1))
        else
            break
        fi
    done
    echo "$(date '+%F %T') $left script(s) left"
    [ "$left" -eq 0 ] && exit 0
    sleep "$pause"
done
