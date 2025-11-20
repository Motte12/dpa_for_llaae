#!/bin/bash
#SBATCH --job-name=dpa-tuning
#SBATCH --partition=clara
#SBATCH --gpus=v100:1
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
epochs=100

# 3.1) Echo for debugging
echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"

srun ~/.conda/envs/dpa/bin/python train_joint_dpa_automated.py \
    --settings_file v2_dpa_train_settings.json \
    --encoder "$enc" \
    --in_dim 648 \
    --latent_dim "$ld" \
    --num_layer "$nln" \
    --hidden_dim "$hdn" \
    --noise_dim_dec "$ndd" \
    --noise_dim_lm "$ndl" \
    --num_layer_lm 2 \
    --hidden_dim_lm "$hdl" \
    --lam "$lambd" \
    --batch_size "$bs" \
    --epochs "$epochs"
