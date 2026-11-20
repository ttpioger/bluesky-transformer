#!/bin/bash
# Submit one long run as a chain of SLURM jobs (2026-11-20). Each job continues the saved state (--resume) and
# stops before its time limit; the next job starts when the previous one ends (afterany).
# A job that finds the run finished exits at once, so extra jobs in the chain cost almost nothing.
#
# Run from bluesky_rl_transformer/:
#   slurm/chain.sh <n_jobs> <run_name> [train arguments ...]
# Extra sbatch options (e.g. a seed array, more CPUs, a time limit) go in SBATCH_OPTS:
#   SBATCH_OPTS="--array=0-2 --time=24:00:00" slurm/chain.sh 6 sa_v2L --steps 50000000 ...
set -euo pipefail
if [ $# -lt 2 ]; then
    echo "usage: [SBATCH_OPTS=...] slurm/chain.sh <n_jobs> <run_name> [train arguments ...]" >&2
    exit 2
fi
N="$1"; shift
mkdir -p logs
PREV=""
for i in $(seq 1 "$N"); do
    DEP=()
    [ -n "$PREV" ] && DEP=(--dependency=afterany:"$PREV")
    # shellcheck disable=SC2086
    PREV=$(sbatch --parsable ${SBATCH_OPTS:-} ${DEP[@]+"${DEP[@]}"} slurm/train.sbatch "$@")
    PREV="${PREV%%;*}"
    echo "job $i/$N: $PREV"
done
