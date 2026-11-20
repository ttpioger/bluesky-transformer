#!/bin/bash
# DelftBlue job (2026-11-20): CPU partitions compute-p1 / compute-p2, no GPU. Only the SLURM options differ from
# slurm/train.sbatch; the job body is that file (resume, wall limit, exit codes, compare, seed-42 test).
#
# Submit from bluesky_rl_transformer/:
#   sbatch slurm/delftblue.sbatch <run_name> [train arguments ...]
#   SBATCH_SCRIPT=slurm/delftblue.sbatch slurm/chain.sh <n_jobs> <run_name> [train arguments ...]
# Options on the sbatch line override the lines below (e.g. --partition=compute-p1 --cpus-per-task=48).
# The venv must be set in ~/.bashrc (UV_PROJECT_ENVIRONMENT); sbatch passes the environment to the job.
#
#SBATCH --job-name=bluesky_train
#SBATCH --account=research-ae-co
#SBATCH --partition=compute-p2
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%x_%j.out

exec bash "$SLURM_SUBMIT_DIR/slurm/train.sbatch" "$@"
