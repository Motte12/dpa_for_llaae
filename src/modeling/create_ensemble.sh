#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dpa # insert the name of your own conda environment here


# configuration
NO_EPOCHS=50 # specify the model you want to use in terms of its training epochs
ENS_MEMBERS=100 # number of ensemble members to generate, code is not robust to any changes of this number

global_settings="../../settings.json"
#MODEL_PATH="../../" # if using pre-trained model from repository root directory
MODEL_PATH=$(jq -r '.paths.output_dir' "$global_settings") # path of the trained model
MODEL="_devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001_pene0" # folder that contains the trained models, adjust if necessary

# specify path to save generated ensemble
save_path="${MODEL_PATH}/${MODEL}" 
echo "savepath: $save_path"
ensemble_save_path_eth="${save_path}/dae_ensemble_after_${NO_EPOCHS}_epochs/"
echo "ensemble save path: $ensemble_save_path_eth"


# specify model
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"


### load model configs ###
cfg="${MODEL_PATH}/${MODEL}/model_and_train_settings.json"

alpha=$(jq -r '.alpha' "$cfg")
batch_norm=$(jq -r '.batch_norm' "$cfg")
batch_size=$(jq -r '.batch_size' "$cfg")
encoder=$(jq -r '.encoder' "$cfg")
epochs=$(jq -r '.epochs' "$cfg")
hidden_dim=$(jq -r '.hidden_dim' "$cfg")
hidden_dim_lm=$(jq -r '.hidden_dim_lm' "$cfg")
in_dim=$(jq -r '.in_dim' "$cfg")
in_dim_lm=$(jq -r '.in_dim_lm' "$cfg")
lam=$(jq -r '.lam' "$cfg")
latent_dim=$(jq -r '.latent_dim' "$cfg")
lr=$(jq -r '.lr' "$cfg")
noise_dim_dec=$(jq -r '.noise_dim_dec' "$cfg")
noise_dim_lm=$(jq -r '.noise_dim_lm' "$cfg")
num_layer=$(jq -r '.num_layer' "$cfg")
num_layer_lm=$(jq -r '.num_layer_lm' "$cfg")
out_activation=$(jq -r '.out_activation // empty' "$cfg")
resblock=$(jq -r '.resblock' "$cfg")
settings_file=$(jq -r '.settings_file' "$cfg")

# create test ensemble
# using interactive slurm
# srun -N1 -n1 python create_test_ensemble.py \
# using no slurm
python create_test_ensemble.py \
    --ens_members $ENS_MEMBERS \
    --save_path_ensemble_single $ensemble_save_path_eth \
    --model_path "$MODEL_PATH/${MODEL}" \
    --encoder_model $ENCODER \
    --decoder_model $DECODER \
    --latent_map_model $LATENT_MAP \
    --no_epochs $NO_EPOCHS \
    --standardize_predictors 1 \
    --autoencode_only 0 \
    --latent_dim $latent_dim \
    --hidden_dim $hidden_dim \
    --num_layers $num_layer \
    --noise_dim_dec $noise_dim_dec \
    --hidden_dim_lm $hidden_dim_lm \
    --noise_dim_lm $noise_dim_lm \
    --lambd $lam \
    --bs $batch_size \
    --bn $batch_norm \
    --settings_file_path $global_settings &
wait