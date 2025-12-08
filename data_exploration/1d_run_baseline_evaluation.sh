#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate dpa

###############
### V1 Data ###
###############

python evaluate_pytorch_quantile_regression.py \
    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v1_data_quantile_regression_ger_gradient_descent_2025-11-26_14-32/ \
    --results_save_path eval_results/ \
    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/dpa_ensemble_after_30_epochs/eth_ensemble_after_30_epochs/ETH_gen_dpa_ens_30_dataset_restored.nc \
    --data_version v1 \
--one_dimensional_ger 1

###############################
### V1 Data counterfactuals ###
###############################

#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v1_data_quantile_regression_ger_gradient_descent_2025-11-26_14-32/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/dpa_ensemble_after_80_epochs/eth_ensemble_after_80_epochs/ETH_cf_gen_dpa_ens_80_dataset_restored.nc \
#    --data_version v1 \
#    --eval_counterfactuals 1



###############
### V3 Data ###
###############

#python evaluate_pytorch_quantile_regression.py \
#    --model_path /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v3_data_quantile_regression_ger_gradient_descent_2025-11-23_19-04/ \
#    --results_save_path eval_results/ \
#    --compare_model /work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v3_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_gen_dpa_ens_20_dataset_restored.nc \
#    --data_version v3