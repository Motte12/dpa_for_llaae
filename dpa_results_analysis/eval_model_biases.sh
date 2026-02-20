#!/bin/bash

dirs=()
for d in /home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/*/; do
    [ -d "$d" ] && dirs+=("$d")
done

printf '%s\n' "${dirs[@]}"


#########################
### for model in dirs ###
#########################

eval "$(conda shell.bash hook)"
conda activate dpa
echo "DPA Environment activated"

# create ensembles
# submit the two slurm job scripts

# === Shared configuration ===
NO_EPOCHS=50
ENS_MEMBERS=100
MODEL_PATH="/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_model/" # ADJUST
MODEL="_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue_lr5e-05_peneFalse" # ADJUST
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"
data_version="v6_dpa_train_settings.json"

# STANDARDIZE PREDICTORS?
# --bn or --no_bn

# save paths
results_save_comment_val="validation_set_reference_period_1950-1980_${data_version}"
ensemble_save_path_val="${MODEL_PATH}${MODEL}/${results_save_comment_val}/dpa_ensemble_after_${NO_EPOCHS}_epochs/"

results_save_comment_eth="eth_test_set_reference_period_1950-1980_${data_version}"
ensemble_save_path_eth="${MODEL_PATH}${MODEL}/${results_save_comment_eth}/dpa_ensemble_after_${NO_EPOCHS}_epochs/"

### load model configs ###
cfg="${MODEL_PATH}${MODEL}/model_and_train_settings.json"

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

##########################

# Validation set ensemble
srun -N1 -n1 python3 create_dpa_ensemble_with_LE_validation_set.py \
    --ens_members $ENS_MEMBERS \
    --save_path_ensemble_single $ensemble_save_path_val \
    --model_path "$MODEL_PATH${MODEL}" \
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
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}" &

# create ETH ensemble
srun -N1 -n1 python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py \
    --ens_members $ENS_MEMBERS \
    --save_path_ensemble_single $ensemble_save_path_eth \
    --model_path "$MODEL_PATH${MODEL}" \
    --encoder_model $ENCODER \
    --decoder_model $DECODER \
    --latent_map_model $LATENT_MAP \
    --no_epochs $NO_EPOCHS \
    --standardize_predictors 1 \
    --autoencode_only 0 \
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}" &

# create ERA5 ensemble
srun -N1 -n1 python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py \
    --ens_members $ENS_MEMBERS \
    --save_path_ensemble_single $ensemble_save_path_eth \
    --model_path "$MODEL_PATH${MODEL}" \
    --encoder_model $ENCODER \
    --decoder_model $DECODER \
    --latent_map_model $LATENT_MAP \
    --no_epochs $NO_EPOCHS \
    --standardize_predictors 1 \
    --autoencode_only 0 \
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}" &

#########################
### run model_bias.py ###
#########################

srun -N1 -n1 python3 model_bias.py \
    --model_path # from above
    --model
    --validation_truth_save_path # insert 
    --eth_cf_truth_save_path # insert
    --eth_fact_truth_save_path # insert
    --era5_fact_truth_save_path # insert
    --era5_cf_truth_save_path # insert
    --eval_no_epochs ${NO_EPOCHS}
    --save_path
    