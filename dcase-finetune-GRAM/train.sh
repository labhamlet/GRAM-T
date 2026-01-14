#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=SELD
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/GRAM-T/seld-dcase2023-GRAM
export HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer


python3 train_seldnet.py
