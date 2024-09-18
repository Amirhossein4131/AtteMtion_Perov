#!/bin/bash 
#SBATCH --job-name=gen-nphase2
#SBATCH --gpus=ampere:1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=16G
#SBATCH --output=gen-nphase2.out


eval "$(conda shell.bash hook)"
source activate base
#1g.10gb:1
srun python3 main.py
