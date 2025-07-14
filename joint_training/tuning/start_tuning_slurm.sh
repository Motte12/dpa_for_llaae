#!/bin/bash
#SBATCH --job-name=dpa-tuning
#SBATCH --array=0-863%24              # Adjust based on how many configs you define
#SBATCH --output=/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output/slurm-%A_%a.out
#SBATCH --partition=clara-long
#SBATCH --gpus=rtx2080ti:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=5-00:00:00


# Define hyperparameters inline using arrays
# model
latent_dims=(10 20 50)
encoder=(learnable PCA)
hidden_dim_NNs=(50 100)
num_layers_NNs=(4 6)
noise_dim_dec=(5 10 20)

# latent map
hidden_dim_lm=(20 50)
noise_dim_lm=(20 100)

#training
lambda=(0 0.5 1)

# Use SLURM_ARRAY_TASK_ID to index into the arrays
ld=${latent_dims[$SLURM_ARRAY_TASK_ID]}
enc=${encoder[$SLURM_ARRAY_TASK_ID]}
hdn=${hidden_dim_NNs[$SLURM_ARRAY_TASK_ID]}
nln=${num_layers_NNs[$SLURM_ARRAY_TASK_ID]}
ndd=${noise_dim_dec[$SLURM_ARRAY_TASK_ID]}

hdl=${hidden_dim_lm[$SLURM_ARRAY_TASK_ID]}
ndl=${noise_dim_lm[$SLURM_ARRAY_TASK_ID]}

lambd=${lambda[$SLURM_ARRAY_TASK_ID]}



~/.conda/envs/dpa/bin/python ../train_joint_dpa_automated.py \
    --settings_file dpa_tune_settings.json \
    --encoder $enc \
    --in_dim 648 \
    --latent_dim $ld \
    --num_layer $nln \
    --hidden_dim $hdn \
    --noise_dim_dec $ndd \
    --noise_dim_lm $ndl \
    --num_layer_lm 2 \
    --hidden_dim_lm $hdl \
    --lam $lambd

# 1) load output dir
ARCHIVE_DIR=$(jq -r '.output_dir' dpa_tuning_settings.json)


# 2) Copy this script (resolved path) into your archive:
cp "$(readlink -f "$0")" "$ARCHIVE_DIR/$(basename "$0")"
