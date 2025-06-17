#!/bin/bash

# load conda module
module load Anaconda3

# Define the conda environment name
CONDA_ENV_NAME="dpa"

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Run the final Python training script
python3 train_dpa_joint_5daily_JJA.py