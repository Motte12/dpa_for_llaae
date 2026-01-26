#!/bin/bash
#SBATCH --job-name=dpa-tuning
#SBATCH --partition=clara
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00



# define hyperparameters
ld=50
enc=learnable
hdn=50
nln=6
ndd=5
hdl=50
ndl=20
lambd=0.5
bs=128
epochs=50

# 3.1) Echo for debugging
echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"

srun ~/.conda/envs/dpa/bin/python train_spatial_StoNet.py \
    --settings_file ../joint_training/v4_dpa_train_settings.json \
    --save_dir "/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/StoNet/v4_spatial/" \
    --num_layer "$nln" \
    --hidden_dim "$hdn" \
    --noise_dim_dec "$ndd" \
    --batch_norm 0 \
    --batch_size "$bs" \
    --epochs "$epochs"
