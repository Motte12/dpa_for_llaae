#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1-01:01
#SBATCH --mem=200G
#SBATCH --partition=clara

eval "$(conda shell.bash hook)"
conda activate dpa
python3 results_sheet_master.py