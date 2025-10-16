## Directories
joint_training --> contains joint training analysis and results

joint_training/tuning/ --> contains tuning results of the joint model

dpa_results_analysis/ --> structured/automated analysis of results

## code

- dpa_ensemble.py -> load trained DPA model and create the DPA ensemble (Factual and Counterfactual)
- evaluation.py -> contains functions to evaluate DPA ensemble
- utils.py -> contains all sort of helper functions


## Tuning
- tuning analysis in "llaae_new/DistributionalPrincipalAutoencoder/joint_training/tuning/analyse_tuning.ipynb"
- started tuning with start_tuning_slurm.sh
	+ loops through hyperparameter combinations
- some tuning jobs weren't executed correctly
	+ their corresponding slurm IDs are in timedout_jobs.txt
	+ these jobs were redone with: repeat_timed_out_jobs_start_tuning_slurm.sh
- tuning results folder:
	+ tuning1: in "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output"
	+ repeated, initially missed: "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output_missed_jobs"

- tuning result/best performing model is in "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_tuning_output/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5" 
	+ retrained model with same parameters is in /work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/

## Folders
/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1 --> contains current model along with predicted DPA ensembles
/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1_noise1 --> contains same model (newly trained) as dpa_model3_tuning1 but with 100 noise dimensions instead of 20

## joint_training/analysing_dpa_results

**Data:**
- input data in: "/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_input_data/"
    + v1_until16102025/ -> first version of trining/test data until 16.10.2025
    + 
- output data in: "/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/" 

**Workflow:**

(combine workflow into a master script)

0. train model ...
    + joint_training/start_training_slurm.sh and joint_training/train_joint_dpa_automated.py
1. load model and create ensemble
    + (first create new directory to save DPA ensemble --> being done automatically)
    + (with load_model_create_ensemble.ipynb)
    + with ETH_test_create_dpa_ensemble_with_ETH_test_set.py, LE_train_create_dpa_ensemble_with_LE_train_set.py and create_dpa_ensemble_with_validation_set.py
    + /work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1_noise1/ --> contains model and predicitons with 100 noise dimensions in the latent map
2. slurm calculations
    + create_results_array.py
    + calculate_energy_score.py
    + create_results_array_zarr.py -> create xarray dataset of DPA ensemble (the full restored one including nans) and save as .zarr
4. Analysis: Analogues and rank histograms
    + with module_dpa_analysis.ipynb
5. Spatial maps, time series, extremes
   + rank_hist_map.ipynb

**Workflow for result analysis combined in dpa_results_analysis/analysis_results_sheet_master.py**


- created DPA ensemble (100 members) with LE train data (first 128 000 time steps) with "llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/create_dpa_ensemble_with_LE_train_set.py"
    + calculate energy "loss with llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/analyse_dpa_ensemble_from_LE_train_set.py" (per map)
    + analyse train error with "llaae_new/DistributionalPrincipalAutoencoder/dpa_results_analysis/LE_train_error.py"
