#!/bin/bash
#SBATCH --job-name=train_dpa              # Job name
#SBATCH --output=/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/my_job_%j.out    # Output file (%j = job ID)
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for serial)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --mem=100G                      # Total memory per node
#SBATCH --time=30:00:00                # Time limit (HH:MM:SS)
#SBATCH --partition=paula            # Partition (queue) name
#SBATCH --gpus=a30:2


# source activate my_env  # If using conda

# Run your command
./start_jointdpa_training.sh