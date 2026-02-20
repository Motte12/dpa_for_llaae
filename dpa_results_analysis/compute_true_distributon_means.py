import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut
import evaluation
from matplotlib.patches import Patch
import torch


def main():
    
    #####################
    ### True Datasets ###
    #####################
    settings_file_path = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v6_dpa_train_settings.json"
    settings_file_path_era5 = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v6_dpa_train_settings_ERA5.json"
    
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)

    with open(settings_file_path_era5, 'r') as file:
        settings_era5 = json.load(file)

    ###############################
    ### Validation Test Factual ###
    ###############################
    
    ds = xr.open_dataset(settings['dataset_trefht'])
    print("Validation data loaded")
    
    # set train/test split
    ds_test = ds.isel(time=slice(4769 * 90, 476900)) #4769 * 80
    

    # weigh array
    #weights = np.cos(np.deg2rad(ds_test.lat))
    #ds_test_weighted = ds_test.TREFHT * weights

    
    # transform to torch tensors
    x_te = ut.data_to_torch(ds_test, "TREFHT")
    
    # remove NaNs from Temperature data
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)
    x_te_reduced
    
    x_te_reduced_mean = x_te_reduced.mean(dim=1) #/ weights.sum().values
    
    # assign arrays
    test = x_te_reduced_mean.detach().cpu().numpy()
    print("Validation test array computed, shape:", test, test.shape)
    sys.exit()
    # save test array
    #np.save(f"{validation_truth_save_path}", test)

    ########################
    ### ETH Test Factual ###
    ########################
    
    eth_trefht_fact = xr.open_dataset(settings["dataset_trefht_eth_transient"])
    print("ETH factual data loaded")
    
    # transform to torch tensors
    x_eth_fact = ut.data_to_torch(eth_trefht_fact, "TREFHT")
    
    # remove NaNs from Temperature data
    x_eth_fact_reduced, mask_x_te = ut.remove_nan_columns(x_eth_fact)
    
    x_eth_fact_reduced_mean = x_eth_fact_reduced.mean(dim=1)
    
    # assign arrays
    x_eth_fact_reduced_mean_numpy = x_eth_fact_reduced_mean.detach().cpu().numpy()
    print("ETH factual test array computed, shape:", x_eth_fact_reduced_mean_numpy.shape)

    # save test array
    #np.save(f"{eth_fact_truth_save_path}", x_eth_fact_reduced_mean_numpy)
    
    ###############################
    ### ETH Test Counterfactual ###
    ###############################
    
    eth_trefht_cf = xr.open_dataset(settings['dataset_trefht_eth_nudged_shifted'])
    print("ETH counterfactual data loaded")

    # transform to torch tensors
    x_eth_cf = ut.data_to_torch(eth_trefht_cf, "TREFHT")
    
    # remove NaNs from Temperature data
    x_eth_cf_reduced, mask_x_te = ut.remove_nan_columns(x_eth_cf)
    
    x_eth_cf_reduced_mean = x_eth_cf_reduced.mean(dim=1)
    
    # assign arrays
    x_eth_cf_reduced_mean_numpy = x_eth_cf_reduced_mean.detach().cpu().numpy()
    print("ETH cf test array computed, shape:", x_eth_fact_reduced_mean_numpy.shape)

    # save test array
    #np.save(f"{eth_cf_truth_save_path}", x_eth_cf_reduced_mean_numpy)
    
    ########################
    ### ERA5 Test Factual ###
    ########################
    
    era5_trefht_fact = xr.open_dataset(settings_era5['dataset_trefht_eth_transient'])
    print("ERA5 factual data loaded")
    # transform to torch tensors
    x_era5_fact = ut.data_to_torch(era5_trefht_fact, "TREFHT")
    
    # remove NaNs from Temperature data
    x_era5_fact_reduced, mask_x_te = ut.remove_nan_columns(x_era5_fact)
    
    x_era5_fact_reduced_mean = x_era5_fact_reduced.mean(dim=1)
    
    # assign arrays
    x_era5_fact_reduced_mean_numpy = x_era5_fact_reduced_mean.detach().cpu().numpy()
    print("ERA5 factual test array computed, shape:", x_era5_fact_reduced_mean_numpy.shape)
    
    
    # save test array
    #np.save(f"{era5_fact_truth_save_path}", x_era5_fact_reduced_mean_numpy)
    
    ###############################
    ### ERA5 Test Counterfactual ###
    ###############################
    
    era5_trefht_cf = xr.open_dataset(settings_era5['dataset_trefht_eth_nudged_shifted'])
    print("ERA5 counterfactual data loaded")
    # transform to torch tensors
    x_era5_cf = ut.data_to_torch(era5_trefht_cf, "TREFHT")
    
    # remove NaNs from Temperature data
    x_era5_cf_reduced, mask_x_te = ut.remove_nan_columns(x_era5_cf)
    
    x_era5_cf_reduced_mean = x_era5_cf_reduced.mean(dim=1)
    
    # assign arrays
    x_era5_cf_reduced_mean_numpy = x_era5_cf_reduced_mean.detach().cpu().numpy()
    print("ERA5 cf test array computed, shape:", x_era5_cf_reduced_mean_numpy.shape)

    # save test array
    #np.save(f"{era5_cf_truth_save_path}", x_era5_cf_reduced_mean_numpy)


if __name__ == "__main__":
    main()