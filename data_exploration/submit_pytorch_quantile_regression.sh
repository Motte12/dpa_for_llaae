#!/bin/bash
#SBATCH --job-name=pt_qu_regr
#SBATCH --cpus-per-task=1
#SBATCH --partition=paula
#SBATCH --gpus=a30:2
#SBATCH --mem=400G
#SBATCH --time=1-5:20

# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"

#eval "$(conda shell.bash hook)"
#conda activate dpa #oder geocat something env

# create directory
# Get current date, hour and minute
timestamp=$(date +"%Y-%m-%d_%H-%M")

# Name prefix
name="/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/quantile_regression_ger_gradient_descent"

# Create directory with timestamp
dirname="${name}_${timestamp}"
mkdir -p "$dirname"

echo "Created directory: $dirname"


# Run the preprocessing script starting from line 72
~/.conda/envs/dpa/bin/python pytorch_quantile_regression.py \
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings.json" \
    --delta 0.00001 \
    --n_epochs 1000 \
    --save_path "$dirname/"

echo "Job finished at $(date)"
