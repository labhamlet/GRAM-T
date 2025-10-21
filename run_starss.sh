#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn24
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.out

cd ~/phd/GRAM-T
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-eval
cd seld-dcase-2022-GRAM

python3 train_seldnet.py 2 > out.txt
