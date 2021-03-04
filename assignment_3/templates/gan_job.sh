#!/bin/bash

#SBATCH --job-name=dl_gan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time 3:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load 2019
module load CUDA/10.0.130
module load Anaconda3/2018.12

source activate dl

srun python3 a3_gan_template.py --device=cuda
