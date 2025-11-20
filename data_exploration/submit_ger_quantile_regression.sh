#!/bin/bash
#SBATCH --job-name=qu_regr_ger
#SBATCH --cpus-per-task=1
#SBATCH --partition=paula
#SBATCH --mem=400G
#SBATCH --time=1-23:20

# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"


# Run the script 
~/.conda/envs/dpa/bin/python quantile_regression_ger.py

echo "Job finished at $(date)"
