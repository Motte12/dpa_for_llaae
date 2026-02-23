#!/bin/bash
#SBATCH --job-name=dpa-train
#SBATCH --partition=clara
#SBATCH --mem=50G
#SBATCH --time=0-08:00:00


# define hyperparameters
ld=100
enc=learnable
hdn=100
nln=6
ndd=100
hdl=50
ndl=20
lambd=0.5
bs=128
epochs=100
learn_rate=0.0001
alpha=1.5
batch_norm=0

# 'latent_dim': 100, 'encoder': 'learnable', 'hidden_dim_NN': 100, 'num_layers_NN': 6, 'noise_dim_dec': 100, 'hidden_dim_lm': 50, 'noise_dim_lm': 20, 'lamb': 0.5, 'epoch': 100, 'batch_norm': 'False', 'alpha': 1.5, 'learn_rate': 0.0001

# 3.1) Echo for debugging
echo "TASK ${SLURM_ARRAY_TASK_ID} → latent_dim=$ld, encoder=$enc, hidden_dim_NN=$hdn, num_layers_NN=$nln, noise_dim_dec=$ndd, hidden_dim_lm=$hdl, noise_dim_lm=$ndl, lambda=$lambd"


~/.conda/envs/dpa/bin/python train_joint_dpa_automated.py \
    --settings_file v5_dpa_train_settings_home.json \
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
    --batch_norm "$batch_norm" \
    --epochs "$epochs" \
    --alpha "$alpha" \
    --include_pen_e 0 \
    --lr "$learn_rate"
