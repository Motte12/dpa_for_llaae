#!/bin/bash
#SBATCH --job-name=analogues
#SBATCH --cpus-per-task=64
#SBATCH --partition=paula
#SBATCH --mem=400G
#SBATCH --time=1-01:20

# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"


# Run the script 
~/.conda/envs/dpa/bin/python create_analogues_parallel_v4_data.py

echo "Job finished at $(date)"
