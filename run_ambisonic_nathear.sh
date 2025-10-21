#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn24
#SBATCH --time=02:00:00
#SBATCH --output=grams/slurm_output_%A_%a.out
#SBATCH --array=0-5


task_names=(
fsd50k-v1.0-full
dcase2016_task2-hear2021-full
# beijing_opera-v1.0-hear2021-full
# esc50-v2.0.0-full
# libricount-v1.0.0-hear2021-full
# speech_commands-v0.0.2-5h
# mridangam_stroke-v1.5-full
# mridangam_tonic-v1.5-full
# tfds_crema_d-1.0.0-full
# nsynth_pitch-v2.2.3-5h
# vox_lingua_top10-hear2021-full
)
tasks_dirs=(
/projects/0/prjs1261/tasks_noisy_ambisonics
/projects/0/prjs1261/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
# /projects/0/prjs1338/tasks_noisy_ambisonics
)

cd ~/phd/GRAM-T
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-eval
cd listen-eval-kit

steps=(50000 100000 200000)
step_idx=$((SLURM_ARRAY_TASK_ID % 3))
task_idx=$((SLURM_ARRAY_TASK_ID / 3))

step=${steps[$step_idx]}
task_name=${task_names[$task_idx]}
tasks_dir=${tasks_dirs[$task_idx]}


embeddings_dir=/projects/0/prjs1338/NoisyEmbeddingsAmbisonicsGRAM$step
score_dir=nathear_ambisonicsgram$step


# # CLEAN
# weights=/gpfs/work5/0/prjs1261/saved_models_naturalistic_mixing/InChannels=1/Fraction=1.0/CleanDataFraction=1.0/Model=GRAM-T/ModelSize=base/LR=0.0002/BatchSize=96/NrSamples=16/Patching=frame/MaskPatch=160/InputL=200/Cluster=False/step=$step.ckpt
# in_channels=1


# AMBISONICS
weights=/gpfs/work5/0/prjs1261/saved_models_naturalistic_mixing_ambisonics/InChannels=7/Fraction=1.0/CleanDataFraction=0.0/Model=GRAM-T/ModelSize=base/LR=0.0002/BatchSize=96/NrSamples=16/Patching=frame/MaskPatch=160/InputL=200/Cluster=False/step=$step.ckpt
in_channels=7

model_name=hear_configs.GRAMT
strategy=raw
use_mwmae_decoder=true
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name

mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/test.predicted-scores.json /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name

rm -r -d -f $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name