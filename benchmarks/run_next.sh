#!/bin/zsh
# Finish what the 2026-08-25 grid was interrupted doing, then extend the k-sweep at N=5.
#
# STAGE A completes the N=6 column at k=1,2,3 -- the shape stage 1 was cut off in the middle of.
# `N6_k1` resumes at starts 1 and 5 (the two the 04:30 handover discarded); `N6_k2` and `N6_k3`
# were cells 5 and 6 of that stage and never ran at all. This is the cheap, high-value half: it
# squares the N=6 column against N=5 and gives the FIRST test at N=6 of the mirror-symmetry
# observation in HANDOFF section 12, which is the evidence a symmetry-reduced search would rest on.
#
# STAGE B fills `N5_k4` and `N5_k5`, which were never scheduled -- stage 1 only ever covered
# k in 1..3. That completes a k-sweep at fixed N, which is what the high-k conjecture needs.
#
# WORKERS=5 AT N<=6 IS THE PROVEN-SAFE CONFIGURATION. It ran 5.5 hours clean on this machine
# overnight. The kernel panic came ~12 minutes into a FOUR-worker N=12 stage, and that tier stays
# off this machine until one N=12 start has been measured under `/usr/bin/time -l`.
#
# Resumable at cell AND start granularity, so killing this at any moment loses only the starts in
# flight, and re-running the same command picks up exactly where it stopped.
set -u
cd /Users/hringer/projects/python/douse
LOG=benchmarks/run/grid_logs
mkdir -p $LOG

echo "=== stage A: N=6 column, k=1,2,3  ($(date '+%H:%M')) ==="
./.venv/bin/python3 -u -m benchmarks.run_grid --N 6 --k 1 2 3 \
    --starts 8 --max-iter 400 --workers 5 2>&1 | tee $LOG/stageA_N6.log

echo "=== stage B: N=5, k=4,5  ($(date '+%H:%M')) ==="
./.venv/bin/python3 -u -m benchmarks.run_grid --N 5 --k 4 5 \
    --starts 8 --max-iter 400 --workers 5 2>&1 | tee $LOG/stageB_N5.log

echo "=== done $(date '+%H:%M') ==="
