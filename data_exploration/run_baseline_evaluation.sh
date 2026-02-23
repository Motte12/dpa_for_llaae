#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate dpa



#####################################
### model5 V6 Data validation set ###
#####################################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v6_data_quantile_regression_ger_gradient_descent_2026-02-13_10-20/ \
#    --qr_epoch 200 \
#    --results_save_path eval_results/validation_v6_data_model5/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_data_model5/_devicecpu100_6_100_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001/model5_validation_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_50_epochs/eth_ensemble_after_50_epochs/ETH_gen_dpa_ens_50_dataset_restored.nc \
#    --data_version v6_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 1

##############################
### model5 V6 Data test set ###
##############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v6_data_quantile_regression_ger_gradient_descent_2026-02-13_10-20/ \
#    --qr_epoch 400 \
#    --results_save_path eval_results/test_v6_data_model5/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_data_model5/_devicecpu100_6_100_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001/model5_eth_test_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_50_epochs/eth_ensemble_after_50_epochs/ETH_gen_dpa_ens_50_dataset_restored.nc \
#    --data_version v6_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0

##############################
### V6 Data validation set ###
##############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v6_data_quantile_regression_ger_gradient_descent_2026-02-13_10-20/ \
#    --qr_epoch 200 \
#    --results_save_path eval_results/validation_v6/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue_lr5e-05_peneFalse/validation_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_50_epochs/eth_ensemble_after_50_epochs/ETH_gen_dpa_ens_50_dataset_restored.nc \
#    --data_version v6_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 1

##############################
### V6 Data test set ###
##############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v6_data_quantile_regression_ger_gradient_descent_2026-02-13_10-20/ \
#    --qr_epoch 200 \
#    --results_save_path eval_results/test_v6/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue_lr5e-05_peneFalse/eth_test_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_50_epochs/eth_ensemble_after_50_epochs/ETH_gen_dpa_ens_50_dataset_restored.nc \
#    --data_version v6_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0

########################################
### V6 Data test set counterfactuals ###
########################################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v6_data_quantile_regression_ger_gradient_descent_2026-02-13_10-20/ \
#    --qr_epoch 200 \
#    --results_save_path eval_results/test_cf_v6/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v6_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue_lr5e-05_peneFalse/eth_test_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_50_epochs/eth_ensemble_after_50_epochs/ETH_cf_gen_dpa_ens_50_dataset_restored.nc \
#    --data_version v6_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0 \
#    --eval_counterfactuals 1

##############################
### V5 Data validation set ###
##############################
python evaluate_pytorch_quantile_regression.py \
    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-02-09_11-52/ \
    --qr_epoch 200 \
    --results_save_path eval_results/validation_v5/ \
    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/_50_4_50_5_1001_100_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisTrue_lr5e-05/validation_set_reference_period_1950-1980_v5_dpa_train_settings.json/dpa_ensemble_after_110_epochs/eth_ensemble_after_110_epochs/ETH_gen_dpa_ens_110_dataset_restored.nc \
    --data_version v5_dpa_train_settings_work.json \
    --one_dimensional_ger 0 \
    --standardize_predictors 1 \
    --eval_validation_set 1

##############################
### V5 Data ERA5 test data ###
##############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-02-09_11-52/ \
#    --qr_epoch 400 \
#    --results_save_path eval_results/Era5/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue/ERA5_reference_period_1950-1980_v5_dpa_train_settings_ERA5.json/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_gen_dpa_ens_20_dataset_restored.nc \
#    --data_version v5_dpa_train_settings_ERA5.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0



#############################
### V5 Data ETH test data ###
#############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-02-09_11-52/ \
#    --qr_epoch 400 \
#    --results_save_path eval_results/Era5_cf/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue/ERA5_reference_period_1950-1980_v5_dpa_train_settings_ERA5.json/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_cf_gen_dpa_ens_20_dataset_restored.nc \
#    --data_version v5_dpa_train_settings_ERA5.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0 \
#    --eval_era5 1 \
#    --eval_counterfactuals 1

#############################
### V5 Data ETH test data ### 21.02.2026
#############################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-02-09_11-52/ \
#    --qr_epoch 200 \
#    --results_save_path eval_results/test_should_be_v5/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue/06022026_reference_period_1950-1980_v5_dpa_train_settings.json/dpa_ensemble_after_30_epochs/eth_ensemble_after_30_epochs/ETH_cf_gen_dpa_ens_30_dataset_restored.nc \
#    --data_version v5_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0 \
#    --eval_era5 0 \
#    --eval_counterfactuals 0


########################################
### V5 Data ETH test counterfactuals ###
########################################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-02-09_11-52/ \
#    --qr_epoch 400 \
#    --results_save_path eval_results/test_cf/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v5_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.0_bs128_bnisTrue/06022026_reference_period_1950-1980_v5_dpa_train_settings.json/dpa_ensemble_after_15_epochs/eth_ensemble_after_15_epochs/ETH_cf_gen_dpa_ens_15_dataset_restored.nc \
#    --data_version v5_dpa_train_settings.json \
#    --one_dimensional_ger 0 \
#    --standardize_predictors 1 \
#    --eval_validation_set 0 \
#    --eval_counterfactuals 1


######################
### StoNet V4 Data ###
######################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/StoNet/v4_data_1d_ger_trained_30_epochs_predictions_dataset.nc \
#    --data_version v4 \
#    --one_dimensional_ger 0


###############
### Analogues / V4 Data ###
###############
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/StoNet/v4_spatial/_6_50_5_bs128_bnisFalse_30_epochs_trained/restored_StoNet_predicted_fact_ensembles_ETH_predictors_standardized_training_and_inference_30_epochs_trained_v4_data.nc \
#    --data_version v4 \
#    --analogues 0 \
#    --one_dimensional_ger 0

###############
### Analogues / V4 Data ###
###############
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/analogues/analogue_ensemble_10pcs_5analogues_100analoguemembers_v4_data_complete.nc \
#    --data_version v4 \
#    --analogues 1 \
#    --one_dimensional_ger 0

###############
### V4 Data ###
###############
python evaluate_pytorch_quantile_regression.py \
    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
    --results_save_path eval_results/v4_test/ \
    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisFalse/dpa_ensemble_after_100_epochs/eth_ensemble_after_100_epochs/ETH_gen_dpa_ens_100_dataset_restored.nc \
    --data_version v4 \
    --one_dimensional_ger 0 \
    --eval_counterfactuals 0


#######################
### Autoencode only ###
#######################
#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/autoencode_only_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisFalse/dpa_ensemble_after_100_epochs/eth_ensemble_after_100_epochs/ETH_gen_dpa_ens_100_dataset_restored.nc \
#    --data_version v4 \
#    --one_dimensional_ger 0


###############
### V1 Data ###
###############

#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v1_data_quantile_regression_ger_gradient_descent_2025-11-26_14-32/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/eth_ensemble_after_30_epochs/ETH_gen_dpa_ens_30_dataset_restored.nc \
#    --data_version v1

###############################
### V4 Data counterfactuals ###
###############################

#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_gen_dpa_ens_20_dataset_restored.nc \
#    --data_version v4 \
#    --eval_counterfactuals 1



###############
### V3 Data ###
###############

#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v3_data_quantile_regression_ger_gradient_descent_2025-11-23_19-04/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v3_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_gen_dpa_ens_20_dataset_restored.nc \
#    --data_version v3