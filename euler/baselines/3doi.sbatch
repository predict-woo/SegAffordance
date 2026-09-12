#!/bin/bash
#SBATCH --job-name=3doi-sf3d
#SBATCH --output=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_rtx_pro_6000:2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
#SBATCH --time=120:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
# 3DOI (SAM ViT-B) on SF3D, released recipe. 2 GPUs is the hard per-user cap of the
# INFK-cvg-leonhard-students QOS (a cli_filter rejects --gpus>2 outright), and RTX PRO 6000
# (96 GB, ~2.5 h queue) beats the A100 80GB (~12 h queue) for this job.
# Checkpoints every epoch and resumes from checkpoints/checkpoint.pth, so the 5-day wall
# clock is not a limit: the job requeues itself and continues.
#   sbatch euler/baselines/3doi.sbatch
set -euo pipefail
export EULER_BASE="${EULER_BASE:-$SCRATCH/baselines}"
SEG="${SEG:-$HOME/SegAffordance}"
mkdir -p "$EULER_BASE/logs"
trap 'echo "$(date -u) caught USR1, requeueing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
echo "== $(date -u) job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
bash "$SEG/euler/baselines/chain.sh" 3doi train &
wait $!
echo "== $(date -u) job $SLURM_JOB_ID finished"
