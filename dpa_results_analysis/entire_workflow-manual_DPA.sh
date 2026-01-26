#!/bin/bash
#SBATCH --job-name=dpa_analysis_120_epochs
#SBATCH --output=/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/slurm_outputs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1-01:01
#SBATCH --mem=200G
#SBATCH --partition=paula


#eval "$(conda shell.bash hook)"
#conda activate dpa
#echo "DPA Environment activated"

# create ensembles
# submit the two slurm job scripts

# === Shared configuration ===
NO_EPOCHS=20
ENS_MEMBERS=100
MODEL_PATH="/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/"
MODEL="_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue"
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"
data_version="v4"

# STANDARDIZE PREDICTORS?
# --bn or --no_bn

# save paths
ensemble_save_path="${MODEL_PATH}${MODEL}/dpa_ensemble_after_${NO_EPOCHS}_epochs/"
results_save_comment="_${data_version}_training_data"

# ETH ensemble
#srun -N1 -n1 python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py \
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
#wait
#echo "Exit ..."
#exit
#wait
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

    ~/.conda/envs/dpa/bin/python analysis_results_sheet_ETH_master_slim.py \
        --period_start $start \
        --period_end $end \
        --ensemble_path $ensemble_save_path \
        --no_epochs $NO_EPOCHS \
        --ens_members $ENS_MEMBERS \
        --calculate_e_loss_per_ti 0 \
        --StoNet_ensemble 0 \
        --save_path_eth "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
        --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
        --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/${data_version}_dpa_train_settings.json" &

done
#--save_path_eth "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
#--save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_${results_save_comment}" \
        
# Wait for all srun jobs to finish before exiting
wait