#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -p JX-GPU-IB
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -J sentencebert
#SBATCH --mem 256g
#SBATCH -o %j-%N-finetune.outs

srun -l python -u finetune_senbert.py

