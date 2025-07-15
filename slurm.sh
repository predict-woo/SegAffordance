#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --gpus=rtx_4090:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

cp -r /cluster/project/cvg/students/andrye/sf3d/data.lmdb /dev/shm/data.lmdb && torchrun --standalone --nnodes=1 --nproc_per_node=2 train_SF3D.py --config config/sf3d_train_full_final.yaml
