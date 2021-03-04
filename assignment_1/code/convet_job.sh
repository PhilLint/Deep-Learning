#!/bin/bash

#SBATCH --job-name=dl_conv_pytorch_DB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time 0:10:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load 2019
module load CUDA/10.0.130
module load Anaconda3/2018.12

source activate dl

srun python3 train_convnet_pytorch.py
