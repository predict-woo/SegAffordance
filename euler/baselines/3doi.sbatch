#!/bin/bash
#SBATCH --job-name=3doi-sf3d
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:40g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
# 3DOI (SAM ViT-B) on SF3D, released recipe (200 epochs). Checkpoints every epoch and resumes
# from checkpoints/checkpoint.pth, so a timeout just requeues.
#   sbatch euler/baselines/3doi.sbatch
set -euo pipefail
BASE="${EULER_BASE:-$SCRATCH/baselines}"
export EULER_BASE="$BASE"
SEG="${SEG:-$HOME/SegAffordance}"
trap 'echo "$(date -u) caught USR1, requeueing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
echo "== $(date -u) job $SLURM_JOB_ID on $(hostname); $(nvidia-smi -L | wc -l) GPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
bash "$SEG/euler/baselines/chain.sh" 3doi train &
wait $!
echo "== $(date -u) job $SLURM_JOB_ID finished"
