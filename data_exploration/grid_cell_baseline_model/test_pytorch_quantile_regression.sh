#!/bin/bash
#SBATCH --job-name=pt_qu_regr
#SBATCH --cpus-per-task=1
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=100G
#SBATCH --time=0-5:20


# create directory
# Get current date, hour and minute
timestamp=$(date +"%Y-%m-%d_%H-%M")

# Name prefix

name="/work2/fl53wumy-llaae_ws_new/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent"

# Create directory with timestamp
dirname="${name}_${timestamp}"
mkdir -p "$dirname"

echo "Created directory: $dirname"


# Run the preprocessing script starting from line 72
~/.conda/envs/dpa/bin/python grid_cells_pytorch_quantile_regression.py \
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v5_dpa_train_settings_home.json" \
    --delta 0.00001 \
    --n_epochs 200 \
    --save_path "$dirname/" \
    --standardize_predictors 1

echo "Job finished at $(date)"
