#!/bin/bash

#python LE_train_create_dpa_ensemble_with_LE_train_set.py \
#    --autoencode_only 1 \
#    --ens_members 100 \
#    --save_path_ensemble_single "/Users/friederl/Documents/EcoN_project/LLAAE/dpa_output_data/autoencoder_only/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/" \
#    --model_path "/Users/friederl/Documents/EcoN_project/LLAAE/dpa_output_data/autoencoder_only/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128" \
#    --encoder_model "model_enc_30.pt" \
#    --decoder_model "model_dec_30.pt" \
#    --latent_map_model "model_pred_30.pt" \
#    --no_epochs 30 \
#    --settings_file_path "/Users/friederl/Documents/EcoN_project/LLAAE/DPA/code/dpa_for_llaae/joint_training/v1_dpa_train_settings_local.json"

python analyse_dpa_ensemble_from_LE_train_set.py \
    --period_start 1850 \
    --period_end 2100 \
    --ens_members 100 \
    --save_path_le "/Users/friederl/Documents/EcoN_project/LLAAE/dpa_output_data/autoencoder_only/" \
    --save_path_eth "/Users/friederl/Documents/EcoN_project/LLAAE/dpa_output_data/autoencoder_only/" \
    --ensemble_path "/Users/friederl/Documents/EcoN_project/LLAAE/dpa_output_data/autoencoder_only/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/" \
    --no_epochs 30 \
    --settings_file_path "/Users/friederl/Documents/EcoN_project/LLAAE/DPA/code/dpa_for_llaae/joint_training/v1_dpa_train_settings_local.json"