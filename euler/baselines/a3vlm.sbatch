#!/bin/bash
#SBATCH --job-name=a3vlm-sf3d
#SBATCH --output=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/baselines/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
# A3VLM (SPHINX-1k 13B) on SF3D.
#
# NOT SUBMITTABLE TODAY (checked 2026-09-12): the QOS INFK-cvg-leonhard-students caps this
# account at 2 GPUs per user and a client-side lua cli_filter rejects the job before a job id
# exists --  "You can only use 2 GPUs as part of INFK-cvg-leonhard-students in the ls_polle
# share." A3VLM needs at least 4 GPUs because `--data_parallel sdp` shards the optimizer over
# DATA-parallel ranks only: with model_parallel_size 2, 2 GPUs give DP=1, no sharding, and
# ~75 GB/GPU of AdamW state (measured OOM on 2 x H100 80GB).
# Keep this file for the day the cap is raised (the peer share ls_grossm already grants 8);
# until then A3VLM runs on RunPod (runpod/baselines/a3vlm/chain.sh).
set -euo pipefail
export EULER_BASE="${EULER_BASE:-$SCRATCH/baselines}"
SEG="${SEG:-$HOME/SegAffordance}"
mkdir -p "$EULER_BASE/logs"
trap 'echo "$(date -u) caught USR1, requeueing $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
echo "== $(date -u) job $SLURM_JOB_ID on $(hostname); $(nvidia-smi -L | wc -l) GPUs"
bash "$SEG/euler/baselines/chain.sh" a3vlm train &
wait $!
