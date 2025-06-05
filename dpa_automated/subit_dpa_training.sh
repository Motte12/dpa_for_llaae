#!/bin/bash
#SBATCH --job-name=my_job              # Job name
#SBATCH --output=logs/output_%j.log    # Output log (%j expands to job ID)
#SBATCH --error=logs/error_%j.err      # Error log
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=8G                       # Total memory
#SBATCH --time=02:00:00                # Time limit hh:mm:ss
#SBATCH --partition=standard           # Partition/queue name
#SBATCH --mail-type=END,FAIL           # Mail on job end or fail
#SBATCH --mail-user=you@example.com    # Your email address

# Load any required modules (optional)
# module load python/3.10

# Activate conda environment (optional)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# Print job info
echo "Running job on $SLURM_JOB_NODELIST with job ID $SLURM_JOB_ID"

# Run your command
python my_script.py
