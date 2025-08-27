##############
### README ###
##############

**Workflow:**

(combine workflow into a master script)

0. train model
    + joint_training/start_training_slurm.sh and joint_training/train_joint_dpa_automated.py
1. load model and create ensemble
    + first create new directory to save DPA ensemble
    + with load_model_create_ensemble.ipynb
2. slurm calculations
    + create_results_array.py
    + calculate_energy_score.py
4. Analysis: Analogues and rank histograms
    + with module_dpa_analysis.ipynb
5. Spatial maps, time series, extremes
   + rank_hist_map.ipynb

