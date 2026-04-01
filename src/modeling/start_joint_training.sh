#!/bin/bash
#SBATCH --job-name=dpa-train
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --mem=50G
#SBATCH --time=0-08:00:00

# use slurm if required

# start training
~/.conda/envs/dpa/bin/python train_joint_dae.py # add arguments for using different settings than in ~/settings.json