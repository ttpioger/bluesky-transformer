#!/bin/bash
# SLURM job for scripts.train (and scripts.compare after it) on the DAIC cluster.
#
# Submit from bluesky_rl_transformer/ (the job uses the submit folder as its work folder):
#   mkdir -p logs
#   sbatch slurm/train.sbatch <run_name> [train arguments ...]
#
# Seeds as a job array (one job per seed; the name gets _s<seed>, --seed is added):
#   sbatch --array=0-2 slurm/train.sbatch sa_v2 [train arguments ...]
#
# Long runs (2026-11-20): the job always adds --resume, so a job continues models/<run_name>_state.pt if it
# exists. It also adds --max-wall-hours (time left in the job minus WALL_MARGIN_MIN, default 30 min) unless you
# give it, so the job saves its state before the SLURM limit. Submit a chain of jobs with slurm/chain.sh.
# Exit codes of scripts.train: 0 finished in this job, 3 stopped at the wall limit (the next job continues),
# 4 already finished before (nothing to do).
# After a finished run: scripts.compare on models/<run_name>.pt (COMPARE=1), then the official competition test
# (seed 42, 1000 episodes) on models/<run_name>_best.pt (TEST=1; needs --val-every in the train arguments).
#
# CPU-only job (no GPU): add --gres=none --partition=<cpu partition> on the sbatch line.
# Options on the sbatch line override the #SBATCH lines below.
# A GPU job gets --device cuda. Give --device cpu in the train arguments to override it.
#
#SBATCH --job-name=bluesky_train
#SBATCH --account=daic-ae-asm-giovan
#SBATCH --partition=all
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch slurm/train.sbatch <run_name> [train arguments ...]" >&2
    exit 2
fi
NAME="$1"; shift

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/train.py ]; then
    echo "Submit from bluesky_rl_transformer/ (scripts/train.py not found in $PWD)." >&2
    exit 2
fi
# The venv can live outside the repo (UV_PROJECT_ENVIRONMENT), e.g. when the repo is on a share that refuses chmod.
PY="${UV_PROJECT_ENVIRONMENT:-../.venv}/bin/python"
if [ ! -x "$PY" ]; then
    echo "No venv at $PY. Run 'uv sync --frozen' in the repo root on the login node first." >&2
    exit 2
fi

SEED_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    NAME="${NAME}_s${SLURM_ARRAY_TASK_ID}"
    SEED_ARGS=(--seed "$SLURM_ARRAY_TASK_ID")
fi

DEVICE=cpu
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
    DEVICE=cuda
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

# One thread per BlueSky worker process; the learner sets its own threads with --learner-threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=..
export SDL_VIDEODRIVER=dummy

echo "job $SLURM_JOB_ID on $(hostname), name $NAME, device $DEVICE, cpus ${SLURM_CPUS_PER_TASK:-?}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

# Hours left in this job minus a margin for the last validation and the state file.
WALL_ARGS=()
if [[ " $* " != *" --max-wall-hours "* ]] && [ -n "${SLURM_JOB_ID:-}" ]; then
    LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)       # [d-]hh:mm:ss, mm:ss or UNLIMITED
    if [[ "$LEFT" =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]]; then
        SEC=$(( 10#${BASH_REMATCH[2]:-0}*86400 + 10#${BASH_REMATCH[4]:-0}*3600 + 10#${BASH_REMATCH[5]}*60 + 10#${BASH_REMATCH[6]} ))
        SEC=$(( SEC - ${WALL_MARGIN_MIN:-30}*60 ))
        if [ "$SEC" -gt 600 ]; then
            WALL_ARGS=(--max-wall-hours "$(awk -v s="$SEC" 'BEGIN{printf "%.3f", s/3600}')")
        fi
    fi
fi
echo "time left: ${LEFT:-?}; ${WALL_ARGS[*]:-no wall limit}"

set +e
"$PY" -m scripts.train --name "$NAME" --device "$DEVICE" --resume ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
    ${WALL_ARGS[@]+"${WALL_ARGS[@]}"} "$@"
RC=$?
set -e
case $RC in
    0) echo "training finished in this job" ;;
    3) echo "stopped at the wall limit; the next job of the chain continues (--resume)"; exit 0 ;;
    4) echo "run already finished; nothing to do"; exit 0 ;;
    *) echo "scripts.train failed with exit code $RC" >&2; exit "$RC" ;;
esac

if [ "${COMPARE:-1}" = "1" ]; then
    "$PY" -m scripts.compare --model "models/$NAME.pt" --workers "${COMPARE_WORKERS:-6}" \
        --wandb "${COMPARE_WANDB:-off}"
fi
if [ "${TEST:-1}" = "1" ] && [ -f "models/${NAME}_best.pt" ]; then
    # official protocol on the snapshot chosen on the validation seeds (never chosen on seed 42)
    "$PY" -m scripts.competition_eval --model "models/${NAME}_best.pt" --workers "${TEST_WORKERS:-8}"
fi
