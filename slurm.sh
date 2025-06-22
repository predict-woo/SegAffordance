#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --gpus=rtx_4090:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

cp -r /cluster/scratch/andrye/SF3D_lmdb/data.lmdb /dev/shm/data.lmdb
python train_SF3D.py --config config/sf3d_train_full.yaml