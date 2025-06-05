#!/bin/bash

# load conda module
module load Anaconda3

# Define the conda environment name
CONDA_ENV_NAME="dpa"

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Define custom output directory
OUTPUT_DIR="./output_results"
SAVE_DIR="./log_directory/"
TRAINING_EPOCHS=5

# Run the final Python training script
python3 train_dpa_final.py --output_dir "$OUTPUT_DIR" --training_epochs "$TRAINING_EPOCHS" --save_dir "$SAVE_DIR"

