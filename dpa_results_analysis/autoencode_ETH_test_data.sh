#!/bin/bash
#SBATCH --job-name=dpa_analysis_120_epochs
#SBATCH --output=/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/slurm_outputs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=0-04:01
#SBATCH --mem=900G
#SBATCH --partition=paula


eval "$(conda shell.bash hook)"
conda activate dpa
echo "DPA Environment activated"

# create ensembles
# submit the two slurm job scripts

# === Shared configuration ===
NO_EPOCHS=20
ENS_MEMBERS=100
MODEL_PATH="/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_autoencoder_only/v2_data/"
MODEL="_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128"
#########
### 0 ###
#########

                                                                                                                                                         
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"

echo "models path"
echo "${MODEL_PATH}${MODEL}"

# save paths
ensemble_save_path="${MODEL_PATH}${MODEL}/dpa_ensemble_after_${NO_EPOCHS}_epochs_only_autoencoder_trained/"
results_save_comment="v2_training_data"          #########
                                                 ### 1 ###
                                                 #########

#############################################
### create DPA ensemble from ETH test set ###
#############################################
#srun -N1 -n1 python3 ETH_test_create_dpa_ensemble_with_ETH_test_set.py \
#    --ens_members $ENS_MEMBERS \
#    --save_path_ensemble_single $ensemble_save_path \
#    --model_path "$MODEL_PATH${MODEL}" \
#    --encoder_model $ENCODER \
#    --decoder_model $DECODER \
#    --latent_map_model $LATENT_MAP \
#    --no_epochs $NO_EPOCHS \
#    --noise_dim_dec 5 \
#    --autoencode_only 1 \
#    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings_post2000.json" &  #########
                                                                                                                                                               ### 2 ###
                                                                                                                                                               #########

#############################################
### create DPA ensemble from LE train set ###
#############################################
srun -N1 -n1 python3 LE_train_create_dpa_ensemble_with_LE_train_set.py \
    --ens_members $ENS_MEMBERS \
    --save_path_ensemble_single $ensemble_save_path \
    --model_path "$MODEL_PATH${MODEL}" \
    --encoder_model $ENCODER \
    --decoder_model $DECODER \
    --latent_map_model $LATENT_MAP \
    --autoencode_only 1 \
    --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings.json" \
    --no_epochs $NO_EPOCHS &  # ← run in background # number of epochs that model was trained for #########
                                                                                                  ### 3 ###
                                                                                                  #########

wait
echo "Ensemble created"


echo "now analyzing"

# run the analysis
# run both analysis scripts (script for LE train set is included at end of analysis_results_sheet_ETH_master.py)
period_start_years=(1850)
period_end_years=(2100)
#########
### 4 ###
#########

##############################################
### Analyse DPA ensemble from ETH test set ###
##############################################

#for i in "${!period_start_years[@]}"; do
#    start=${period_start_years[$i]}
#    end=${period_end_years[$i]}

#    echo "Running analysis for period ${start}-${end}"
#    echo "Epochs: ${NO_EPOCHS}"
#
#    srun -N1 -n1 python3 analysis_results_sheet_ETH_master.py \
#        --period_start $start \
#        --period_end $end \
#        --ensemble_path $ensemble_save_path \
#        --no_epochs $NO_EPOCHS \
#        --ens_members $ENS_MEMBERS \
#        --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}" \
#        --save_path_eth "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}" \
#        --include_train_analysis 0 \
#        --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings_post2000.json" &
#done

                                                                                                                                                                  #########
                                                                                                                                                                   ### 5 ###
                                                                                                                                                                  #########

##############################################
### Analyse DPA ensemble from LE train set ###
##############################################

for i in "${!period_start_years[@]}"; do
    start=${period_start_years[$i]}
    end=${period_end_years[$i]}

    echo "Running analysis for period ${start}-${end}"
    echo "Epochs: ${NO_EPOCHS}"

    srun -N1 -n1 python3 analyse_dpa_ensemble_from_LE_train_set.py \
        --period_start $start \
        --period_end $end \
        --ensemble_path $ensemble_save_path \
        --no_epochs $NO_EPOCHS \
        --ens_members $ENS_MEMBERS \
        --save_path_eth "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}" \
        --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}" \
        --settings_file_path "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings.json" &

    echo "script submitted"

done
                                                                                                                                                                  #########
                                                                                                                                                                  ### 6 ###
                                                                                                                                                                  #########
                                                                                                                                                                  
                                                                                                                                                                  #########
                                                                                                                                                                  ### 7 ###
                                                                                                                                                                  #########
                                                                                                                                                                  
# Wait for all srun jobs to finish before exiting
#wait

# create summary
#echo "Creating summary"
#srun -N1 -n1 python3 create_summary_page.py \
#    --save_path_eth "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}/period_2000_2100" \
#    --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}" \
#    --save_path "ETH_analysis_results/final_analysis_test_ETH/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_autoencode_only_${results_save_comment}/period_2000_2100" \
#    --period "1850-2100" \
#    --summary_save_comment "AE_only_training_data2000-2100"
                                                        #########
                                                        ### 8 ###
                                                        #########

