#!/bin/bash
#SBATCH -p slurm_me759
#SBATCH --job-name=prob02
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1 
#SBATCH -o prob.o

export CUDA_PROFILE=1
cd /home/sharmila/ece759_project/vj_cuda5/
./vj
