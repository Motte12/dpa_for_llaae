#!/bin/bash
#SBATCH --job-name=dpa-train
#SBATCH --partition=clara
#SBATCH --mem=50G
#SBATCH --time=0-06:00:00
#SBATCH --gpus=v100:1



# alpha=1.5, batch_norm=True, encoder=learnable, hidden_dim_NN=50, hidden_dim_Im=50, lamb=0.5, latent_dim=50, noise_dim_dec=5, noise_dim_Im=100, num_layers_NN=4
# define hyperparameters
ld=100
enc=learnable
hdn=100
nln=6
ndd=5
hdl=50
ndl=20
lambd=0.5
bs=128
epochs=200
learn_rate=0.0001
alpha=1.5

# 3.1) Echo for debugging
echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"


~/.conda/envs/dpa/bin/python train_joint_dpa_automated.py \
    --settings_file v6_dpa_train_settings_save_in_home.json \
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
    --batch_norm 0 \
    --epochs "$epochs" \
    --alpha "$alpha" \
    --include_pen_e 0 \
    --lr "$learn_rate"
