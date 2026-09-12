#!/bin/bash
#SBATCH --job-name=3doi-sf3d
#SBATCH --output=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_rtx_pro_6000:2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
# 3DOI (SAM ViT-B) on SF3D, released recipe. 2 GPUs is the hard per-user cap of the
# INFK-cvg-leonhard-students QOS (a cli_filter rejects --gpus>2 outright), and RTX PRO 6000
# (96 GB, ~2.5 h queue) beats the A100 80GB (~12 h queue) for this job.
# 24 h rather than the 5-day maximum on purpose: the .120h partitions are heavily contended
# (est. start ~9 h vs ~1.6 h for .24h), and the job checkpoints every epoch and requeues itself
# on timeout, so a shorter slot costs only an extra queue wait every 24 h.
#   sbatch euler/baselines/3doi.sbatch
set -euo pipefail
export EULER_BASE="${EULER_BASE:-$SCRATCH/baselines}"
SEG="${SEG:-$HOME/SegAffordance}"
mkdir -p "$EULER_BASE/logs"
trap 'echo "$(date -u) caught USR1, requeueing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
echo "== $(date -u) job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
EPOCHS="${EPOCHS:-62}" bash "$SEG/euler/baselines/chain.sh" 3doi train &
wait $!
echo "== $(date -u) job $SLURM_JOB_ID finished"
