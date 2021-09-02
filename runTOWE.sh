#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ToweTask
#SBATCH --partition=small
#SBATCH --output=results/towe_output.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.mensah@sheffield.ac.uk


module load python/anaconda3
module load cuda/10.2
module load gcc/9.1.0
module load pytorch/1.9.0
source activate pytorch
python run1.py
