# GRAM-T : General-purpose audio representation model Transformer

This repository contains the implementation of GRAM-T, a state-of-the-art audio processing model trained on AudioSet with naturalistic scenes. It includes the listen-eval-kit, an extended fork of hear-eval-kit with enhanced functionality for sound localization tasks. The framework leverages PyTorch Lightning and Hydra with TensorBoard logging for comprehensive hyperparameter optimization.

## System Requirements

This repository has been validated with:
- Python 3.9
- PyTorch 2.5.1

## Installation

### Training Environment

```bash
# Create and activate conda environment
conda create -n gram-t python=3.9 -y
conda activate gram-t

# Install GRAM-M specific dependencies
pip install -r requirements.txt

```

### Evaluation Environment

```bash
conda create -n gram-t-eval python=3.9 -y
conda activate gram-t-eval

# Install GRAM-M specific dependencies
pip install -r requirements_eval.txt
```

## Model Training

### GRAM-T-Time Model Training

```bash
python3 train.py data=audioset data.sr=32000 patching=time data.mask_patch=80 trainer.batch_size=32 trainer.steps=200000
```

### GRAM-T-Patch Model Training

```bash
python3 train.py data=audioset data.sr=32000 patching=frame data.mask_patch=100 trainer.batch_size=32 trainer.steps=200000
```

**Note:** Training requires approximately 72 hours on an A100 GPU. Models and TensorBoard logs are saved to the directory specified in `cfg.save_dir`.

## Downstream Evaluation

### HEAR Benchmark Evaluation

#### Prerequisites

1. **Dataset Preparation:**
   - Follow the instructions at https://hearbenchmark.com/hear-tasks.html to acquire the data
   - For convenience, download pre-processed 32000 Hz data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/)
   - Extract all files to a designated directory (`$TASKS_DIR`)

2. **Pre-trained Model Weights:**
   - Install the latest version of the huggingface with a pip install huggingface. Note: You should install huggingface after setting up the requirements.txt
   - Then, to get the pretrained model weights, navigate to the huggingface directory.
   - Run the gramt_binaural.py. This is the pretrained model, and the definition can be copied across the files. The most important point is to have RunTimeGRAMT discoverable via sys.


#### Feature Extraction and Evaluation

```bash
cd listen-eval-kit

# Define environment variables
embeddings_dir=/path/to/save/embeddings
tasks_dir=$TASKS_DIR
task_name=dcase2016_task2-hear2021-full

# Set model parameters
weights=$MODEL_DIR
model_name=hear_configs.GRAMT #GRAMTTime if Time based model
strategy=raw
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

# Extract features
python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 

# Execute task evaluation
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
```

### Nat-HEAR Benchmark Evaluation

#### Prerequisites

1. **Dataset Preparation:**
   - Access the Nat-HEAR dataset from [repository link]
   - Extract all files to a designated directory (`$NATURALISTIC_TASKS_DIR`)

2. **Pre-trained Model Weights:**
   - Download pre-trained weights from [repository link]
   - Set the path to the weights directory as `$MODEL_DIR`

#### Feature Extraction and Evaluation

```bash
cd listen-eval-kit

# Define environment variables
embeddings_dir=/path/to/save/embeddings
tasks_dir=$NATURALISTIC_TASKS_DIR
task_name=dcase2016_task2-hear2021-full

# Set model parameters
weights=$MODEL_DIR
model_name=hear_configs.GRAMT #GRAMTTime if Time based model
strategy=raw
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

# Extract features
python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 

# Execute task evaluation
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name
```

### Sound Localization on Nat-HEAR Benchmark

#### Prerequisites

1. **Dataset Preparation:**
   - Access the Nat-HEAR-localization dataset from [repository link]
   - Extract all files to a designated directory (`$LOCALIZATION_TASKS_DIR`)

2. **Pre-trained Model Weights:**
   - Download pre-trained weights from [repository link]
   - Set the path to the weights directory as `$MODEL_DIR`

#### Feature Extraction and Evaluation

```bash
cd listen-eval-kit

# Define environment variables
embeddings_dir=/path/to/save/embeddings
tasks_dir=$LOCALIZATION_TASKS_DIR
task_name=dcase2016_task2-hear2021-full

# Set model parameters
weights=$MODEL_DIR
model_name=hear_configs.GRAMT #GRAMTTime if Time based model
strategy=mean
use_mwmae_decoder=true
in_channels=2
model_options="{\"strategy\": \"$strategy\",\"use_mwmae_decoder\": \"$use_mwmae_decoder\", \"in_channels\": $in_channels}"

# Extract features
python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 

# Execute localization evaluation
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy-use-mwmae-decoder=$use_mwmae_decoder-in-channels=$in_channels/$task_name --localization cartesian-regression
```
