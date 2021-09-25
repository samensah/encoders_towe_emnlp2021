#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ToweTask
#SBATCH --partition=small
#SBATCH --output=shell_log/towe_output.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.mensah@sheffield.ac.uk


module load python/anaconda3
module load cuda/10.2
module load gcc/9.1.0
module load pytorch/1.9.0
source activate pytorch

declare -a dataset=("14lap" "14res" "15res" "16res")

# number of runs
for rnum in {1..5}; do
    # loop over datasets
    for data in "${dataset[@]}"; do
        # number of gcn layers
        for layer_num in {0..5}; do
            savedir="run${rnum}_layers${layer_num}_${data}"
            echo "python train.py --dataset ${data} --gcn_layers ${layer_num} --save_dir ${savedir}"
            python train.py --dataset ${data} --gcn_layers ${layer_num} --save_dir ${savedir}
        done
    done
done
