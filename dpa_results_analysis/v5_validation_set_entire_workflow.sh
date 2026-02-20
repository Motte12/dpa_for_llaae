#!/bin/bash


eval "$(conda shell.bash hook)"
conda activate dpa
echo "DPA Environment activated"

# create ensembles
# submit the two slurm job scripts

# === Shared configuration ===
NO_EPOCHS=110
ENS_MEMBERS=100
MODEL_PATH="/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/"
MODEL="_50_4_50_5_1001_100_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisTrue_lr5e-05"
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"
data_version="v5_dpa_train_settings.json"

# STANDARDIZE PREDICTORS?
# --bn or --no_bn

# save paths
results_save_comment="validation_set_reference_period_1950-1980_${data_version}"
ensemble_save_path="${MODEL_PATH}${MODEL}/${results_save_comment}/dpa_ensemble_after_${NO_EPOCHS}_epochs/"

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
    --save_path_ensemble_single $ensemble_save_path \
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


# ETH ensemble
#python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py \
#    --ens_members $ENS_MEMBERS \
#    --save_path_ensemble_single $ensemble_save_path \
#    --model_path "$MODEL_PATH${MODEL}" \
#    --encoder_model $ENCODER \
#    --decoder_model $DECODER \
#    --latent_map_model $LATENT_MAP \
#    --no_epochs $NO_EPOCHS \
#    --standardize_predictors 1 \
#    --autoencode_only 0 \
#    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}" &

# LE train data set ensemble
#srun -N1 -n1 python3 LE_train_create_dpa_ensemble_with_LE_train_set.py \
#    --ens_members $ENS_MEMBERS \
#    --save_path_ensemble_single $ensemble_save_path \
#    --model_path "$MODEL_PATH${MODEL}" \
#    --encoder_model $ENCODER \
#    --decoder_model $DECODER \
#    --latent_map_model $LATENT_MAP \
#    --no_epochs $NO_EPOCHS \
#    --standardize_predictors 1 \
#    --no_bn \
#    --autoencode_only 1 \
#    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}_dpa_train_settings.json" &

#echo "Exit ..."
#exit
wait
echo "Ensemble created, now analyzing"




# run the analysis
# run both analysis scripts (script for LE train set is included at end of analysis_results_sheet_ETH_master.py)
period_start_years=(1850)
period_end_years=(2100)

for i in "${!period_start_years[@]}"; do
    start=${period_start_years[$i]}
    end=${period_end_years[$i]}

    echo "Running analysis for period ${start}-${end}"
    echo "Epochs: ${NO_EPOCHS}"

    srun python3 analysis_results_sheet_LE_validation_set_master_slim.py \
        --period_start $start \
        --period_end $end \
        --ensemble_path $ensemble_save_path \
        --no_epochs $NO_EPOCHS \
        --ens_members $ENS_MEMBERS \
        --calculate_e_loss_per_ti 0 \
        --StoNet_ensemble 0 \
        --save_path_eth "ETH_analysis_results/final_analysis_validation_LE/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
        --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
        --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}" \
        --no_test_members 10 \
        --include_train_analysis 0 &

done

# Wait for all srun jobs to finish before exiting
wait