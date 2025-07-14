#!/bin/bash

#source ~/.conda/etc/profile.d/conda.sh
#eval "$(conda shell.bash hook)"
#conda init
#conda activate dpa


~/.conda/envs/dpa/bin/python ../train_joint_dpa_automated.py \
    --settings_file dpa_tune_settings.json \
    --encoder PCA \
    --in_dim 648 \
    --latent_dim 20 \
    --num_layer 6 \
    --hidden_dim 100 \
    --noise_dim_dec 20 \
    --noise_dim_lm 20 \
    --num_layer_lm 2 \
    --hidden_dim_lm 50

