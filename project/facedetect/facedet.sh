#!/bin/bash

TOP = /home/vinayg/red_ninjas/project
BIN = $TOP/bin


#SBATCH -p slurm_me759
#SBATCH --job-name=facedetect
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1 
#SBATCH -o img.log

export CUDA_PROFILE=1

cp $BIN/facedetect $TOP/facedetect
cd $TOP/facedetect
./facedetect group.pgm img.log
rm -rf facedetect

