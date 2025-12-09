#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

cp -r /cluster/work/cvg/students/andrye/sf3d_processed/data.lmdb /dev/shm && python train_SF3D_better.py fit --config config/sf3d_train.yaml