#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=01:00:00
#SBATCH --output=localization_hear/slurm_output_%A_%a.out
#SBATCH --array=0-3


cd ~/phd/GRAM-T
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-eval
cd listen-eval-kit



tasks=(
    sc_reverb_noisy
)

lambdas=(0.0 0.25 0.5 0.75)
lambda_idx=$((SLURM_ARRAY_TASK_ID % 4))
task_idx=$SLURM_ARRAY_TASK_ID

lambda=${lambdas[$lambda_idx]}
task_name=esc50-v2.0.0-full


embeddings_dir=/projects/0/prjs1338/LocalizationEmbeddingsBinaural$lambda
score_dir=hear_scores_localization_binaural_patch$lambda
tasks_dir=/projects/0/prjs1338/create_nat_hear/tasks_spatial

# # #Binaural
weights=/projects/0/prjs1261/saved_models_naturalistic_mixing/InChannels=2/Fraction=1.0/CleanDataFraction=$lambda/Model=GRAM-T/ModelSize=base/LR=0.0002/BatchSize=96/NrSamples=16/Patching=frame/MaskPatch=160/InputL=200/Cluster=False/step=500000.ckpt
in_channels=2

model_name=hear_configs.GRAMT
strategy=cls
use_mwmae_decoder=true
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name --localization cartesian-regression

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name

mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/test.predicted-scores-localization.json /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/*predictions-localization.pkl /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name

rm -r -d -f $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
