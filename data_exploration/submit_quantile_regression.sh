#!/bin/bash
#SBATCH --job-name=qu_regr
#SBATCH --cpus-per-task=100
#SBATCH --partition=paula
#SBATCH --mem=400G
#SBATCH --time=1-5:20

# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"

#eval "$(conda shell.bash hook)"
#conda activate dpa #oder geocat something env

# Run the preprocessing script starting from line 72
~/.conda/envs/dpa/bin/python quantile_regression_with_saving.py

echo "Job finished at $(date)"
