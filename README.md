This repository contains code to reproduce the results for the extended abstract *Towards a distributional autoencoder for climate counterfactuals* submitted to the Climate Informatics 2026 conference.

## Repository Structure

Towards-a-distributional-autoencoder-for-climate-counterfactuals/
├── README.md
├── LICENSE             
├── src/
│   ├── modeling/                   -> Core modeling code
│   │   ├── __init__.py
│   │   ├── create_ensemble.sh      -> bash script to properly start create_test_ensemble.py
|   |   ├── create_test_ensemble.py -> python script to create an ensemble from a trained model
│   │   ├── pca_encoder.py          -> contains a pca encoder
│   │   ├── start_joint_training.sh -> start python script for training DAE
|   |   └── train_joint_dae.py      -> script to train model
│   ├── analysis/                   -> Code for analysis of model output
│   │   ├── __init__.py
│   │   └── extended_abstract_figure.ipynb -> notebook to create figure in extended abstract
│   └── utils/                      -> Helper functions shared between modeling and analysis
│       ├── __init__.py
│       ├── utils.py
│       ├── dpa_ensemble.py
│       └── evaluation.py
├── environment.yml                 -> Conda environment file 
└──_devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001_pene0 -> pre-trained model

## Instructions


### Data setup
- get the training and test data from **zenodo link**
- create a data directory (arbitrary name) and put the data there (don't change names of the datasets)
- insert the data directory name into `settings.json` in ['paths']['data']
- adjust the paths in settings.json

### Workflow to reproduce the extended abstract figure

1. create a conda environement using the environment.yaml file ([explained here](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))
2. Train the model (or skip this, directly go to step 2 and use the pretrained model in _devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0)
   - start model training by executing start_joint_training.sh
3. Create an ensemble
   - in `create_ensemble.sh`, adjust
       - `MODEL=` and `MODEL_PATH=` accordingly
       - the conda envrionment name in line 5 to the name of your conda environment
   - optional
       - adjust location for saving the generated ensemble `save_path=` (default is in the model directory)
       - adjust the last command (around line 56) if you want to use slurm
   - execute `create_ensemble.sh` to create the ensemble (potentially need to make it executable before `chmod +x create_ensemble.sh`)
4. Analysis with `extended_abstract_figure.ipynb`
   - potentially adjust `dae_ensemble_fact` and `dae_ensemble_cf`
   - run the notebook


This project is **work in progress**. If you encounter any issues or have suggestions, please reach out.

This project builds is based on the [engression framework](https://github.com/xwshen51/engression).