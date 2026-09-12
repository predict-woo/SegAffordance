#!/bin/bash
#SBATCH --job-name=a3vlm-sf3d
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
# A3VLM (SPHINX-1k 13B) on SF3D, released recipe. Resumes from the newest epoch dir, so a
# wall-clock timeout is harmless: the job requeues itself and picks up where it stopped.
#   sbatch euler/baselines/a3vlm.sbatch
# Override the GPU count with: sbatch --gpus=N ...  (N must be >= 4; see chain.sh's DP guard)
set -euo pipefail
BASE="${EULER_BASE:-$SCRATCH/baselines}"
export EULER_BASE="$BASE"
SEG="${SEG:-$HOME/SegAffordance}"
trap 'echo "$(date -u) caught USR1, requeueing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
echo "== $(date -u) job $SLURM_JOB_ID on $(hostname); $(nvidia-smi -L | wc -l) GPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
bash "$SEG/euler/baselines/chain.sh" a3vlm train &
wait $!
echo "== $(date -u) job $SLURM_JOB_ID finished"
