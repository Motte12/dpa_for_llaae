#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1-01:01
#SBATCH --mem=200G
#SBATCH --partition=clara

eval "$(conda shell.bash hook)"
conda activate dpa
echo "DPA Environment activated"
python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py