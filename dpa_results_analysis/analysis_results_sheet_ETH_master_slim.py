import torch
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss, energy_loss_two_sample

import xarray as xr
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.manifold import TSNE
import numpy as np
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import shutil
from datetime import datetime
import argparse

import sys
import os
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import dpa_ensemble as de
import utils as ut
import evaluation
import analyse_dpa_ensemble_from_LE_train_set
import create_summary_page

def log_print(log_path, message):
    print(message)  # print to console
    with open(log_path, "a") as f:
        print(message, file=f)  # also write to file    


def main():
    parser = argparse.ArgumentParser(description="Example script with arguments")
    
    parser.add_argument("--include_train_analysis", type=int, default=0, help="Whether to include analysis of train data.")
    parser.add_argument("--period_start", type=int, help="Start year of period to analyse")
    parser.add_argument("--period_end", type=int, help="End year of period to analyse")
    parser.add_argument("--ensemble_path", type=str, help="Path of DPA ensemble")
    parser.add_argument("--no_epochs", type=int, help="Number of epochs model was trained used for creating this DPA ensemble")
    parser.add_argument("--ens_members", type=int, default=100, help="Number of members in DPA ensemble")
    parser.add_argument("--save_path_le", type=str, help="Save path of LE train set analysis figures")
    parser.add_argument("--save_path_eth", type=str, help="Save path of ETH set analysis figures")
    parser.add_argument("--settings_file_path", type=str, help="Path of settings (datasets) to create ensemble.")
    parser.add_argument("--no_test_members", type=int, default=3, help="Number of members in the test set.")
    parser.add_argument("--calculate_e_loss_per_ti", type=int, default=1, help="Whether to calculate energy loss per time step.")
    parser.add_argument("--StoNet_ensemble", type=int, default=0, help="Whether to evaluate StoNet ensemble.")


    args = parser.parse_args()
    
    time_period = [str(args.period_start), str(args.period_end)]
    
    no_epochs = args.no_epochs
    
    ensemble_path = f"{args.ensemble_path}eth_ensemble_after_{no_epochs}_epochs"

    # save path
    if args.save_path_le is not None:
        print("save path LE is given")
        save_path_le = args.save_path_le
    else:
        print("save path LE is not given")
        save_path_le = f"ETH_analysis_results/final_analysis_train_LE/model_trained_for_{args.no_epochs}_epochs"
        

    if args.save_path_eth is not None:
        print("save path eth is given")
        save_path_eth = f"{args.save_path_eth}/period_{time_period[0]}_{time_period[1]}"
    else:
        print("save path eth is not given")
        save_path_eth = f"ETH_analysis_results/final_analysis_test_ETH/model_trained_for_{args.no_epochs}_epochs/period_{time_period[0]}_{time_period[1]}"

        

    print("save path ETH analysis results:", save_path_eth)
    print("save path LE analysis results:", save_path_le)
    print("include LE train analysis:", args.include_train_analysis)
    print("ensemble load path:", ensemble_path)
    
    
    os.makedirs(save_path_eth, exist_ok=True)
    os.makedirs(save_path_le, exist_ok=True)

    
    log_file = f"{save_path_eth}/test_log_metrics_{time_period[0]}-{time_period[1]}.txt"
    
    # Get current time and print it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(log_file, f"=== Current Time: {current_time} ===")
    log_print(log_file, f"=== Quantiles ===")
    #log_print(log_file, f"\n")


    print(f"{save_path_eth}/Germany")
    print(f"{save_path_eth}/Spain")
    
    # create germany and spain subdirs––±
    os.makedirs(f"{save_path_eth}/Germany", exist_ok=True)
    os.makedirs(f"{save_path_eth}/Spain", exist_ok=True)
    os.makedirs(f"{save_path_eth}/quantiles", exist_ok=True)
    
    
 
    # plotting settings
    title_fontsize = 18
    figsize_map = (10,8)
    figsize_ts = (10,8)
    figsize_hist = (8,6)

    #input_dim = (32, 32)


    # years for time series plotting
    years = ["2000", "2020", "2040", "2060"]#, "2000", "2020", "2040", "2060"]
    # subset to years that are contained in time period!!
    
    # Convert to integers
    start, end = map(int, time_period)
    
    # Filter years that fall within the range
    years = [y for y in map(int, years) if start <= y <= end]
    
    #print("years contained", years)
    
    ens_members=args.ens_members

    
    #################
    ### Load Data ###
    #################
    # create figures page
    #fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # 2x2 grid of subplots

    # load test data
    #print("Loading test data ...")
    
    # Large Ensemble Data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced, pi_period_mean = de.load_test_data(args.settings_file_path)
    print(ds)
    #print("x_te_reduced shape:", x_te_reduced.shape)

    # ETH Ensemble Test data
    z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf, _, _ = de.load_eth_test_data(args.settings_file_path)
    # z500                  -> test predictors
    # mask_x_te_eth_fact    -> land mask
    # ds_test_eth_fact      -> factual test temperatures (xarray dataset) lat: 32, lon: 32, time: 14307
    # x_te_reduced_eth_fact -> land grid cells factual temperature data
    # x_te_reduced_eth_cf   -> land grid cells counterfactual temperature data

    print("x_te_reduced_eth_fact:", x_te_reduced_eth_fact.shape)
    
    slice_end_index = int(x_te_reduced_eth_fact.shape[0]/args.no_test_members)
    print("Slice end index:", slice_end_index)
    # datasets
    ds_test_1300_eth_fact = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1])) # HERE TP
    #print("ds_test_1300_eth_fact:", ds_test_1300_eth_fact)
    ds_test_1300_eth_cf = ds_test_eth_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1])) # HERE TP

    # get indices of time slices
    time_index = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).get_index("time")
    #print("Time index:", time_index)
    indices = time_index.get_indexer(ds_test_1300_eth_fact.time.values)
    start_idx, end_idx = indices[0], indices[-1]+1 # add 1 to include last index
    start_idx_1400, end_idx_1400 = end_idx, 2*end_idx
    start_idx_1500, end_idx_1500 = 2*end_idx, 3*end_idx
    
    
    #if time_period[-1] == "2100":
    #    end_idx = indices[-1]
    
    #############################

    print("Start index:", start_idx)
    print("End index:", end_idx)

    print("Start index 1400:", start_idx_1400)
    print("End index 1400:", end_idx_1400)

    print("Start index 1500:", start_idx_1500)
    print("End index 1500:", end_idx_1500)

    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[start_idx])
    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[end_idx])

    
    #################
    ### Test Data ###
    #################
    
    # PYTORCH arrays
    # Factual Test/True temperatures
    eth_fact_1300_test_reduced = x_te_reduced_eth_fact[:slice_end_index,:][start_idx:end_idx,:] # HERE
    eth_fact_1400_test_reduced = x_te_reduced_eth_fact[slice_end_index:2*slice_end_index,:][start_idx:end_idx,:]
    eth_fact_1500_test_reduced = x_te_reduced_eth_fact[-slice_end_index:14307,:][start_idx:end_idx,:]
    print("eth_fact_1300_test_reduced shape:", eth_fact_1300_test_reduced.shape)
    mask_x_te = mask_x_te_eth_fact

    # Counterfactual
    # Factual Test/True temperatures
    eth_cf_1300_test_reduced = x_te_reduced_eth_cf[:slice_end_index,:][start_idx:end_idx,:] # HERE
    eth_cf_1400_test_reduced = x_te_reduced_eth_cf[slice_end_index:2*slice_end_index,:][start_idx:end_idx,:]
    eth_cf_1500_test_reduced = x_te_reduced_eth_cf[-slice_end_index:14307,:][start_idx:end_idx,:]
    print("eth_fact_1300_test_reduced counterfactual shape:", eth_cf_1300_test_reduced.shape)
    
    #########################
    ### Load DPA Ensemble ###
    #########################
    
    # including nan's
    #dpa_ensemble = xr.open_zarr(f"{save_path_ensemble_single}/dpa_ens_100_dataset_restored.zarr", consolidated=True)
    
    # RAW ensemble without NaNs
    # shape: ensemble_member: 100time: 64000lat_x_lon: 648
    print("Loading DPA ensemble ...")
    print("DPA ensemble load paths:")

    # FACTUAL 
    # shape: ensemble_member: 100, time: 14307, lat_x_lon: 648
    print(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
    print(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    print(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
    print(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    
    if bool(args.StoNet_ensemble):
        # factual

        #actual StoNet
        dpa_ensemble_fact_restored = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/StoNet/v4_spatial/_6_50_5_bs128_bnisFalse_30_epochs_trained/restored_StoNet_predicted_fact_ensembles_ETH_predictors_standardized_training_and_inference_30_epochs_trained_v4_data.nc")
        
        # analogues
        #dpa_ensemble_fact_restored = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/analogues/analogue_ensemble_10pcs_5analogues_100analoguemembers_v4_data_complete.nc").ensemble_temp.transpose("analogue_ensemble_member","time","lat","lon").to_dataset(name="TREFHT").rename_dims({"analogue_ensemble_member": "ensemble_member"}) #.rename({"ensemble_temp": "TREFHT"})
        print("dpa_ensemble_fact_restored", dpa_ensemble_fact_restored)
        print(dpa_ensemble_fact_restored.TREFHT.isnull().any().item())
        print("Mask_x_te shape:", type(mask_x_te), mask_x_te.shape)
        print(mask_x_te)
        print(dpa_ensemble_fact_restored)
        
        ### create raw ###
        ds_flat = dpa_ensemble_fact_restored.stack(latxlon=("lat", "lon"))
        print(ds_flat)
        mask_np = mask_x_te.cpu().numpy().astype(bool)
        dpa_ensemble_fact_raw = ds_flat.isel(latxlon=mask_np)
        print(dpa_ensemble_fact_raw)
        # dims become: (ensemble_member, time, space_selected)

        print(type(x_tr_reduced))
        
        # counterfactual
        
        # actual StoNet
        dpa_ensemble_restored_cf = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/StoNet/v4_spatial/_6_50_5_bs128_bnisFalse_30_epochs_trained/restored_StoNet_predicted_cf_ensembles_ETH_predictors_standardized_training_and_inference_30_epochs_trained_v4_data.nc")

        #### only when using analogues: ###
        #dpa_ensemble_restored_cf = dpa_ensemble_fact_restored
        ####################################
        print(dpa_ensemble_restored_cf)

        ### create raw counterfactual ###
        ds_flat_cf = dpa_ensemble_restored_cf.stack(latxlon=("lat", "lon"))
        print(ds_flat_cf)
        mask_np = mask_x_te.cpu().numpy().astype(bool)
        dpa_ensemble_raw_cf = ds_flat_cf.isel(latxlon=mask_np)
        print(dpa_ensemble_fact_raw)
        
        
    else:
        # factual
        dpa_ensemble_fact_raw = xr.open_dataset(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
        dpa_ensemble_fact_restored = xr.open_dataset(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
        
        # counterfactual
        dpa_ensemble_raw_cf = xr.open_dataset(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
        dpa_ensemble_restored_cf = xr.open_dataset(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")

    # FACTUAL
    dpa_1300_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    #print("dpa_1300_fact_raw:", dpa_1300_fact_raw)
    

    # shape: ensemble_member: 100, time: 14307, lat: 32, lon: 32
    dpa_1300_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    

    # COUNTERFACTUAL
    dpa_1300_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))

    dpa_1300_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    
    
    #############
    ### Tests ###
    #############
    
    

    #################
    ### Quantiles ### add for random locations?? add counterfactual??
    #################
    
    def c_quantiles(test_data, ensemble, quantiles): # in temperature space
        # compute per grid-cell quantiles
        ## in test set
        quantiles_fact_true = torch.quantile(test_data, q=quantiles, dim=0) # shape: (quantiles, features) or for 1d:

        ## in predicted ensemble, mean of quantiles across all ensemble members
        if quantiles.numel() == 1:
            print("===1D quantiles===")
            quantiles_fact_dpa = torch.mean(torch.quantile(ensemble, q=quantiles, dim=1), dim=0) # shape: (quantiles, features)
        
        else:
            quantiles_fact_dpa = torch.mean(torch.quantile(ensemble, q=quantiles, dim=1), dim=1) # shape: (quantiles, ensemble_members, features)

        
        # mean absolute error per grid cell, across quantiles
        if quantiles.numel() == 1:
            qu_gc_diff = torch.abs(quantiles_fact_true - quantiles_fact_dpa)
        else:
            qu_gc_diff = torch.mean(torch.abs(quantiles_fact_true - quantiles_fact_dpa), dim=0)
        
        # spatial mean of (mean absolute error per grid-cell across quantiles)
        mae_spatial_mean = torch.mean(qu_gc_diff)
        
        return qu_gc_diff, mae_spatial_mean 

    quantiles = torch.linspace(0,1,21)

    ###############
    ### Factual ###
    ###############
    test_all_quantiles = []
    test_099_quantile = []
    # 1300
    dpa_1300_fact_raw_pt = torch.from_numpy(dpa_1300_fact_raw.values)
    qu_gc_diff1300, mae_spatial_mean1300 = c_quantiles(eth_fact_1300_test_reduced, dpa_1300_fact_raw_pt, quantiles)
    qu_gc_diff1300_099, mae_spatial_mean1300_099 = c_quantiles(eth_fact_1300_test_reduced, dpa_1300_fact_raw_pt, torch.tensor(0.99))
    test_all_quantiles.append(mae_spatial_mean1300)
    test_099_quantile.append(mae_spatial_mean1300_099)
    print("Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1300)
    print("Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1300_099)

    if args.no_test_members > 1:
        # 1400
        dpa_1400_fact_raw_pt = torch.from_numpy(dpa_1400_fact_raw.values)
        qu_gc_diff1400, mae_spatial_mean1400 = c_quantiles(eth_fact_1400_test_reduced, dpa_1400_fact_raw_pt, quantiles)
        qu_gc_diff1400_099, mae_spatial_mean1400_099 = c_quantiles(eth_fact_1400_test_reduced, dpa_1400_fact_raw_pt, torch.tensor(0.99))
        test_all_quantiles.append(mae_spatial_mean1400)
        test_099_quantile.append(mae_spatial_mean1400_099)
        print("Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1400)
        print("Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1400_099)
    
    
        # 1500
        dpa_1500_fact_raw_pt = torch.from_numpy(dpa_1500_fact_raw.values)
        qu_gc_diff1500, mae_spatial_mean1500 = c_quantiles(eth_fact_1500_test_reduced, dpa_1500_fact_raw_pt, quantiles)
        qu_gc_diff1500_099, mae_spatial_mean1500_099 = c_quantiles(eth_fact_1500_test_reduced, dpa_1500_fact_raw_pt, torch.tensor(0.99))
        test_all_quantiles.append(mae_spatial_mean1500)
        test_099_quantile.append(mae_spatial_mean1500_099)
        print("Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1500)
        print("Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1500_099)

    # compute mean values
    quantile_099_mean = np.mean(test_099_quantile)
    all_quantiles_mean = np.mean(test_all_quantiles)
    print("099 quantile mean MAE across test members:", quantile_099_mean)
    print("all quantiles mean MAE across test members:", all_quantiles_mean)

    log_print(log_file, f"all quantiles mean MAE across test members (ETH 1300,1400,1500): {all_quantiles_mean}")
    log_print(log_file, f"099 quantile mean MAE across test members (ETH 1300,1400,1500): {quantile_099_mean}")

    ######################
    ### Counterfactual ###
    ######################
    
    test_all_quantiles_cf = []
    test_099_quantile_cf = []
    # 1300
    
    dpa_1300_cf_raw_pt = torch.from_numpy(dpa_1300_cf_raw.values)
    qu_gc_diff1300_cf, mae_spatial_mean1300_cf = c_quantiles(eth_cf_1300_test_reduced, dpa_1300_cf_raw_pt, quantiles)
    qu_gc_diff1300_099_cf, mae_spatial_mean1300_099_cf = c_quantiles(eth_cf_1300_test_reduced, dpa_1300_cf_raw_pt, torch.tensor(0.99))
    test_all_quantiles_cf.append(mae_spatial_mean1300_cf)
    test_099_quantile_cf.append(mae_spatial_mean1300_099_cf)
    print("counterfactual Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1300_cf)
    print("counterfactual Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1300_099_cf)

    if args.no_test_members > 1:
        # 1400
        dpa_1400_cf_raw_pt = torch.from_numpy(dpa_1400_cf_raw.values)
        qu_gc_diff1400_cf, mae_spatial_mean1400_cf = c_quantiles(eth_cf_1400_test_reduced, dpa_1400_cf_raw_pt, quantiles)
        qu_gc_diff1400_099_cf, mae_spatial_mean1400_099_cf = c_quantiles(eth_cf_1400_test_reduced, dpa_1400_cf_raw_pt, torch.tensor(0.99))
        test_all_quantiles_cf.append(mae_spatial_mean1400_cf)
        test_099_quantile_cf.append(mae_spatial_mean1400_099_cf)
        print("counterfactual Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1400_cf)
        print("counterfactual Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1400_099_cf)
    
    
        # 1500
        #dpa_1500_cf_raw_pt = torch.from_numpy(dpa_1500_cf_raw.values)
        #qu_gc_diff1500_cf, mae_spatial_mean1500_cf = c_quantiles(eth_cf_1500_test_reduced, dpa_1500_cf_raw_pt, quantiles)
        #qu_gc_diff1500_099_cf, mae_spatial_mean1500_099_cf = c_quantiles(eth_cf_1500_test_reduced, dpa_1500_cf_raw_pt, torch.tensor(0.99))
        #test_all_quantiles_cf.append(mae_spatial_mean1500_cf)
        #test_099_quantile_cf.append(mae_spatial_mean1500_099_cf)
        #print("counterfactual Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean1500_cf)
        #print("counterfactual Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean1500_099_cf)

    # compute mean values
    quantile_099_mean_cf = np.mean(test_099_quantile_cf)
    all_quantiles_mean_cf = np.mean(test_all_quantiles_cf)
    print("counterfactual 099 quantile mean MAE across test members:", quantile_099_mean_cf)
    print("counterfactual all quantiles mean MAE across test members:", all_quantiles_mean_cf)

    log_print(log_file, f"counterfactual, all quantiles mean MAE across test members (ETH 1300,1400,1500): {all_quantiles_mean_cf}")
    log_print(log_file, f"counterfactual, 099 quantile mean MAE across test members (ETH 1300,1400,1500): {quantile_099_mean_cf}")


    #############################
    ### Conditional coverages ###
    #############################
    
    # y_test_np: test truth grid cell time data
    # quantile_predictions_dpa: shape: (14307, 100), np.quantile(trefht_dpa_trans_ger_mean.values.T, np.linspace(0.05, 0.95, 19), axis=1).T
    # quantiles: 
    # dpa_1300_fact_raw: shape (100, 4769, 648)
    # trefht_dpa_trans_ger_mean.values.T.shape: (14307, 100)
    
    quantiles_cq = torch.linspace(0.05, 0.95, 19)
    quantiles_cq_np = np.linspace(0.05, 0.95, 19)


    ### Factual ###
    mae_means = []
    mae099_means = []

    through = zip([eth_fact_1300_test_reduced], [dpa_1300_fact_raw])
    if args.no_test_members > 1:
        through = zip([eth_fact_1300_test_reduced, eth_fact_1400_test_reduced, eth_fact_1500_test_reduced], [dpa_1300_fact_raw, dpa_1400_fact_raw, dpa_1500_fact_raw])
    for y_test_np, dpa_xxxx_fact_raw in through:
        mae_list = []
        mae099_list = []
        for i in range(648):
            print(i)
            y_test_np #= eth_fact_1300_test_reduced[:,i]
            quantile_predictions_dpa = np.quantile(dpa_xxxx_fact_raw[:,:,i].T, np.linspace(0.05, 0.95, 19), axis=1).T
            
            
        
            print("y_test_np.shape", y_test_np.shape)
            print("quantile_predictions_dpa.shape", quantile_predictions_dpa.shape)
            cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)
            # compute MAE between cover_dpa and quantiles_cq
    
    
            mae = np.mean(np.abs(quantiles_cq_np - cover_dpa))
            mae_list.append(mae)
            mae099 = np.abs(quantiles_cq_np[-1] - cover_dpa[-1])
            mae099_list.append(mae099)
        spat_mean_mae = np.mean(mae_list) 
        spat_mean_099 = np.mean(mae099_list)
        mae_means.append(spat_mean_mae)
        mae099_means.append(spat_mean_099)
    
    log_print(log_file, f"coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae_means)}")
    log_print(log_file, f"095 coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae099_means)}")

    ### Counterfactual ###
    mae_means_cf = []
    mae099_means_cf = []
    
    through_cf = zip([eth_cf_1300_test_reduced], [dpa_1300_cf_raw])
    if args.no_test_members > 1:
        through_cf = zip([eth_cf_1300_test_reduced, eth_cf_1400_test_reduced, eth_cf_1500_test_reduced], [dpa_1300_cf_raw, dpa_1400_cf_raw, dpa_1500_cf_raw])
    for y_test_np, dpa_xxxx_cf_raw in through_cf:
        mae_list = []
        mae099_list = []
        for i in range(648):
            print(i)
            y_test_np #= eth_fact_1300_test_reduced[:,i]
            quantile_predictions_dpa = np.quantile(dpa_xxxx_cf_raw[:,:,i].T, np.linspace(0.05, 0.95, 19), axis=1).T
            
            #print("dpa_1300_fact_raw.shape", dpa_1300_fact_raw.shape, type(dpa_1300_fact_raw))
            #print("eth_fact_1300_test_reduced", eth_fact_1300_test_reduced.shape, type(eth_fact_1300_test_reduced))
        
            print("y_test_np.shape", y_test_np.shape)
            print("quantile_predictions_dpa.shape", quantile_predictions_dpa.shape)
            cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)
            # compute MAE between cover_dpa and quantiles_cq
    
    
            mae = np.mean(np.abs(quantiles_cq_np - cover_dpa))
            mae_list.append(mae)
            mae099 = np.abs(quantiles_cq_np[-1] - cover_dpa[-1])
            mae099_list.append(mae099)
        spat_mean_mae = np.mean(mae_list) 
        spat_mean_099 = np.mean(mae099_list)
        mae_means_cf.append(spat_mean_mae)
        mae099_means_cf.append(spat_mean_099)
    
    log_print(log_file, f"counterfactual coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae_means_cf)}")
    log_print(log_file, f"counterfactual 095 coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae099_means_cf)}")
    

    ########################
    ### Autocorrelations ###
    ########################
    # calculate autocorrelations

    ### Factual ###
    ###############
    
    ## of DPA ENSEMBLE
    # dpa ensemble into torch array
    dpa_1300_fact_raw_pt = torch.from_numpy(dpa_1300_fact_raw.values)
    # compute autocorrelation
    autocorr_mult, _ = evaluation.compute_autocorrelation(dpa_1300_fact_raw_pt, mask_x_te, ds_test_eth_fact)

    ## of true TEST data
    autocorr, autocorr_xr = evaluation.compute_autocorrelation(eth_fact_1300_test_reduced, mask_x_te, ds_test_eth_fact)
    
    fig, ax = ut.plot_map(autocorr_xr, np.linspace(0,1,21), cmap="YlOrRd", cmap_label = "Autocorrelation")
    fig.savefig(f"{save_path_eth}/autocorrelation_truth.png")



    # difference between ensemble members and truth ###
    autocorr_diff = autocorr_mult - autocorr
    #print("autocorrelation differene shape:", autocorr_diff.shape)
    autocorr_diff_ens_mean = autocorr_diff.mean(dim=0)
    autocorr_diff_ens_mean_spat_mean = autocorr_diff_ens_mean.mean()
    log_print(log_file, f"Spatial mean factual autocorrelation difference: {autocorr_diff_ens_mean_spat_mean}")
    ###################################################

    autocorr_restored = ut.restore_nan_columns(autocorr_diff_ens_mean, mask_x_te)
    #print("autocorr_restored shape", autocorr_restored.shape)
    autocorr_xr = ut.torch_to_dataarray(autocorr_restored, ds_test_eth_fact)
    fig, ax = ut.plot_map(autocorr_xr, np.linspace(-0.2,0.2,17), cmap="PuOr", cmap_label = "Autocorrelation")
    fig.savefig(f"{save_path_eth}/autocorrelation_diff_ens_mean.png")

    ### Counterfactual ###
    ######################
    ## of DPA ENSEMBLE
    # dpa ensemble into torch array
    dpa_1300_cf_raw_pt = torch.from_numpy(dpa_1300_cf_raw.values)
    # compute autocorrelation
    autocorr_mult_cf, _ = evaluation.compute_autocorrelation(dpa_1300_cf_raw_pt, mask_x_te, ds_test_eth_cf)

    ## of true TEST data
    autocorr_cf, autocorr_xr_cf = evaluation.compute_autocorrelation(eth_cf_1300_test_reduced, mask_x_te, ds_test_eth_cf)
    
    fig, ax = ut.plot_map(autocorr_xr_cf, np.linspace(0,1,21), cmap="YlOrRd", cmap_label = "Autocorrelation")
    fig.savefig(f"{save_path_eth}/autocorrelation_truth_cf.png")



    # difference between ensemble members and truth ###
    autocorr_diff_cf = autocorr_mult_cf - autocorr_cf
    #print("autocorrelation differene shape:", autocorr_diff_cf.shape)
    autocorr_diff_ens_mean_cf = autocorr_diff_cf.mean(dim=0)
    autocorr_diff_ens_mean_spat_mean_cf = autocorr_diff_ens_mean_cf.mean()
    log_print(log_file, f"Spatial mean counterfactual autocorrelation difference: {autocorr_diff_ens_mean_spat_mean_cf}")
    ###################################################

    autocorr_restored_cf = ut.restore_nan_columns(autocorr_diff_ens_mean_cf, mask_x_te)
    #print("autocorr_restored shape", autocorr_restored_cf.shape)
    autocorr_xr_cf = ut.torch_to_dataarray(autocorr_restored_cf, ds_test_eth_cf)
    fig, ax = ut.plot_map(autocorr_xr_cf, np.linspace(-0.2,0.2,17), cmap="PuOr", cmap_label = "Autocorrelation")
    fig.savefig(f"{save_path_eth}/autocorrelation_diff_ens_mean_cf.png")

    ####################
    ### Energy Score ###
    ####################

    ### PER GRID CELL ###
    #####################

    def energy_loss_per_gc(eth_fact, dpa_list, start_idx, end_idx, mask_x_te):
        
        print(f"Now calculating spatial energy loss for indices {start_idx}-{end_idx}")
        e_loss_array = torch.zeros(3,648)
        for i in range(648):
            # keep only subset of tensors in list
            dpa_list_subset = [t[start_idx:end_idx, i] for t in dpa_list] #[t[:4769, i] for t in dpa_list]
            
            ### calculate energy score ###
            e_loss = energy_loss(eth_fact[:,i], dpa_list_subset)
            e_loss_array[0,i] = e_loss[0].detach() 
            e_loss_array[1,i] = e_loss[1].detach()
            e_loss_array[2,i] = e_loss[2].detach()
            #print(i)
            #print("Energy Loss:", e_loss)
    
        print("E Loss shape:", e_loss_array.shape)
    
        return e_loss_array

    
    
    ### Factual function ###
    ########################

    # dpa ensemble for one grid cell list[(time steps,1), (time steps,1), ...]
    # this is the reconstructed array containing 1024 grid cells (many with NaNs)
    if bool(args.StoNet_ensemble):
        ### transform into dpa_list ###
        # ds is your xarray.Dataset
        da = dpa_ensemble_fact_raw["TREFHT"]  # (ensemble_member, time, latxlon)
        
        # Optional but recommended: make sure it's a numeric time axis (not object dtype)
        # (skip if you already have proper datetime64)
        # da = da.assign_coords(time=xr.conventions.decode_cf_datetime(da.time, ...))  # if needed
        
        # Create list[torch.Tensor], each tensor shape: (time, latxlon)
        dpa_list = [
            torch.from_numpy(da.sel(ensemble_member=em).values)   # numpy array (time, latxlon)
            for em in da.ensemble_member.values
        ]
        
        # Example: check shapes
        print(len(dpa_list), dpa_list[0].shape, dpa_list[0].dtype)

        #####
    else:
        _, dpa_list, _, _, _ = ut.load_dpa_arrays(path=f"{ensemble_path}/",
                                      mask = mask_x_te,
                                      ds_coords = ds_test_eth_fact,
                                      ens_members=ens_members)
    
    print("DPA tensor list factual:", len(dpa_list))
    print("DPA tensor list factual element shape:", dpa_list[0].shape) # list elements contain all timesteps 14307 (3 x 4769)
    
    e_loss_array_1300 = energy_loss_per_gc(eth_fact_1300_test_reduced, dpa_list, start_idx, end_idx, mask_x_te)
    if args.no_test_members > 1:
        e_loss_array_1400 = energy_loss_per_gc(eth_fact_1400_test_reduced, dpa_list, start_idx_1400, end_idx_1400, mask_x_te)
        e_loss_array_1500 = energy_loss_per_gc(eth_fact_1500_test_reduced, dpa_list, start_idx_1500, end_idx_1500, mask_x_te)

    
    e_loss_test_mean = np.mean([e_loss_array_1300[0,:].mean()])
    s1_loss_test_mean = np.mean([e_loss_array_1300[1,:].mean()])
    s2_loss_test_mean = np.mean([e_loss_array_1300[2,:].mean()])
    
    if args.no_test_members > 1:
        e_loss_test_mean = np.mean([e_loss_array_1300[0,:].mean(), e_loss_array_1400[0,:].mean(), e_loss_array_1500[0,:].mean()])
        s1_loss_test_mean = np.mean([e_loss_array_1300[1,:].mean(), e_loss_array_1400[1,:].mean(), e_loss_array_1500[1,:].mean()])
        s2_loss_test_mean = np.mean([e_loss_array_1300[2,:].mean(), e_loss_array_1400[2,:].mean(), e_loss_array_1500[2,:].mean()])
    
    log_print(log_file, f"Factual Energy-Score spatial mean across test members: {e_loss_test_mean}")
    log_print(log_file, f"Factual S1-Score spatial mean across test members: {s1_loss_test_mean}")
    log_print(log_file, f"Factual S2-Score spatial mean across test members: {s2_loss_test_mean}")
    
    ### Factual ###
    ###############

    

    
    #print("Now calculating spatial energy loss ...")
    #e_loss_array = torch.zeros(3,648)
    #for i in range(648):
    #    # keep only subset of tensors in list
    #    dpa_list_subset = [t[start_idx:end_idx, i] for t in dpa_list] #[t[:4769, i] for t in dpa_list]
        
    #    ### calculate energy score ###
    #    e_loss = energy_loss(eth_fact_1300_test_reduced[:,i], dpa_list_subset)
    #    e_loss_array[0,i] = e_loss[0].detach() 
    #    e_loss_array[1,i] = e_loss[1].detach()
    #    e_loss_array[2,i] = e_loss[2].detach()
    
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array_1300[0,:].unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(e_loss_array_1300[1,:].unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(e_loss_array_1300[2,:].unsqueeze(0), mask_x_te)

    
    # write to logfile
    # was only 1300
    #log_print(log_file, f"Factual Energy-Score spatial mean: {e_loss_array[0,:].mean()}")
    #log_print(log_file, f"Factual S1-Score spatial mean: {e_loss_array[1,:].mean()}")
    #log_print(log_file, f"Factual S2-Score spatial mean: {e_loss_array[2,:].mean()}")
    
    # transform e_loss_array into xarray
    e_loss_xr = ut.torch_to_dataarray(x_tensor = e_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="energy_loss_total") # add oth dimensin 
    s1_loss_xr = ut.torch_to_dataarray(x_tensor = s1_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="s1_loss")
    s2_loss_xr = ut.torch_to_dataarray(x_tensor = s2_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="s2_loss")
    #print("E loss xr:", e_loss_xr)
    
    #energy_levels = np.linspace(0.4, 1.1, 7)
    #levels = np.linspace(0.8, 2.0, 13)
    energy_levels = np.linspace(0.0, 3.0, 11)
    levels = np.linspace(0.0, 3.0, 11)
    
    # plot energy score
    fig, ax = ut.plot_map(e_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Energy Loss")
    ax.set_title("Factual Energy Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/energy_loss_map.png")
    plt.show()

    # plot S1 loss
    fig, ax = ut.plot_map(s1_loss_xr, levels, cmap="YlOrRd", cmap_label = "Reconstruction Loss (S1)")
    ax.set_title("Factual S1 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/S1_loss_map.png")
    plt.show()

    # plot s2 loss
    fig, ax = ut.plot_map(s2_loss_xr, levels, cmap="YlOrRd", cmap_label = "Variability Loss (S2)")
    ax.set_title("Factual S2 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/S2_loss_map.png")
    plt.show()

    ### Counterfactual function ###
    ###############################

    # dpa ensemble for one grid cell list[(time steps,1), (time steps,1), ...]
    # this is the reconstructed array containing 1024 grid cells (many with NaNs)
    if bool(args.StoNet_ensemble):
        ### transform into dpa_list ###
        # ds is your xarray.Dataset
        da = dpa_ensemble_raw_cf["TREFHT"]  # (ensemble_member, time, latxlon)
        
        # Optional but recommended: make sure it's a numeric time axis (not object dtype)
        # (skip if you already have proper datetime64)
        # da = da.assign_coords(time=xr.conventions.decode_cf_datetime(da.time, ...))  # if needed
        
        # Create list[torch.Tensor], each tensor shape: (time, latxlon)
        dpa_list_cf = [
            torch.from_numpy(da.sel(ensemble_member=em).values)   # numpy array (time, latxlon)
            for em in da.ensemble_member.values
        ]
        
        # Example: check shapes
        print(len(dpa_list), dpa_list[0].shape, dpa_list[0].dtype)

        #####
    else:
        _, dpa_list_cf_pre, _, _, _ = ut.load_both_dpa_arrays(path=f"{ensemble_path}/",
                                      mask = mask_x_te,
                                      ds_coords = ds_test_eth_fact,
                                      ens_members=ens_members,
                                      climate_list = ["cf_gen"]
                                      ) 
        dpa_list_cf = dpa_list_cf_pre[0]

    print("DPA tensor list cf:", len(dpa_list_cf))
    print("DPA tensor list cf element shape:", dpa_list_cf[0].shape)
    
    e_loss_array_1300_cf = energy_loss_per_gc(eth_cf_1300_test_reduced, dpa_list_cf, start_idx, end_idx, mask_x_te)
    
    if args.no_test_members > 1:
        e_loss_array_1400_cf = energy_loss_per_gc(eth_cf_1400_test_reduced, dpa_list_cf, start_idx_1400, end_idx_1400, mask_x_te)
        e_loss_array_1500_cf = energy_loss_per_gc(eth_cf_1500_test_reduced, dpa_list_cf, start_idx_1500, end_idx_1500, mask_x_te)


    e_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[0,:].mean()])
    s1_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[1,:].mean()])
    s2_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[2,:].mean()])
    
    if args.no_test_members > 1:
        e_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[0,:].mean(), e_loss_array_1400_cf[0,:].mean(), e_loss_array_1500_cf[0,:].mean()])
        s1_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[1,:].mean(), e_loss_array_1400_cf[1,:].mean(), e_loss_array_1500_cf[1,:].mean()])
        s2_loss_test_mean_cf = np.mean([e_loss_array_1300_cf[2,:].mean(), e_loss_array_1400_cf[2,:].mean(), e_loss_array_1500_cf[2,:].mean()])
    
    log_print(log_file, f"Counterfactual Energy-Score spatial mean across test members: {e_loss_test_mean_cf}")
    log_print(log_file, f"Counterfactual S1-Score spatial mean across test members: {s1_loss_test_mean_cf}")
    log_print(log_file, f"Counterfactual S2-Score spatial mean across test members: {s2_loss_test_mean_cf}")
    
    
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array_1300_cf[0,:].unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(e_loss_array_1300_cf[1,:].unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(e_loss_array_1300_cf[2,:].unsqueeze(0), mask_x_te)
    
    # transform e_loss_array into xarray
    e_loss_xr = ut.torch_to_dataarray(x_tensor = e_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="energy_loss_total") # add oth dimensin 
    s1_loss_xr = ut.torch_to_dataarray(x_tensor = s1_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="s1_loss")
    s2_loss_xr = ut.torch_to_dataarray(x_tensor = s2_loss_restored, coords_ds = ds_test_eth_fact, lat_dim=32, lon_dim=32, name="s2_loss")
    #print("E loss xr:", e_loss_xr)
    
    
    
    # plot energy score
    fig, ax = ut.plot_map(e_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Energy Loss")
    ax.set_title("Counterfactual Energy Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/cf_energy_loss_map.png")
    plt.show()

    # plot S1 loss
    fig, ax = ut.plot_map(s1_loss_xr, levels, cmap="YlOrRd", cmap_label = "Reconstruction Loss (S1)")
    ax.set_title("Counterfactual S1 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/cf_S1_loss_map.png")
    plt.show()

    # plot s2 loss
    fig, ax = ut.plot_map(s2_loss_xr, levels, cmap="YlOrRd", cmap_label = "Variability Loss (S2)")
    ax.set_title("Counterfactual S2 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/cf_S2_loss_map.png")
    plt.show()

    
    ### PER TIME STEP ###
    #####################

    ### Factual ###
    ###############

    
    ### Counterfactual ###
    ######################

    
    
    
    
    ###########################
    ### PEARSON CORRELATION ###
    ###########################
    # mean of restored factual DPA ensemble
    #dpa_ens_mean_restored = dpa_ensemble_fact_restored.TREFHT.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1300_restored = dpa_1300_fact_restored.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_restored = dpa_1400_fact_restored.mean(dim="ensemble_member")
        dpa_ens_mean_fact_1500_restored = dpa_1500_fact_restored.mean(dim="ensemble_member")

    # mean of restored counterfactual DPA ensemble
    dpa_ens_mean_cf_1300_restored = dpa_1300_cf_restored.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_restored = dpa_1400_cf_restored.mean(dim="ensemble_member")
        dpa_ens_mean_cf_1500_restored = dpa_1500_cf_restored.mean(dim="ensemble_member")

    # mean of raw factual ensemble
    dpa_ens_mean_fact_1300_raw = dpa_1300_fact_raw.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_raw = dpa_1400_fact_raw.mean(dim="ensemble_member")
        dpa_ens_mean_fact_1500_raw = dpa_1500_fact_raw.mean(dim="ensemble_member")

    # mean of raw counterfactual ensemble
    dpa_ens_mean_cf_1300_raw = dpa_1300_cf_raw.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_raw = dpa_1400_cf_raw.mean(dim="ensemble_member")
        dpa_ens_mean_cf_1500_raw = dpa_1500_cf_raw.mean(dim="ensemble_member")

    ### Factual ###
    ###############

    # mean of RAW factual ensemble
    #print("Calculating ensemble mean ...")
    #dpa_fact_1300_raw_mean = dpa_ens_mean_fact_1300_raw.mean(dim="ensemble_member")

    ## turn dpa ens (mean) into torch array
    dpa_ens_mean_fact_1300_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1300_raw.values) #dpa_ens_mean_pt
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1400_raw.values)
        dpa_ens_mean_fact_1500_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1500_raw.values)
    # check shapes and types
    #print("dpa_ens_mean_pt shape", dpa_ens_mean_fact_1300_raw_pt.shape)

    # calculate correlation
    #print("Now calculating pearson correlation")
    # test temperature vs. eg dpa_ens_mean_1300
    r_cols_1300 = evaluation.pearsonr_cols(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)  # shape: (648,)
    if args.no_test_members > 1:
        r_cols_1400 = evaluation.pearsonr_cols(eth_fact_1400_test_reduced, dpa_ens_mean_fact_1400_raw_pt, dim=0)
        r_cols_1500 = evaluation.pearsonr_cols(eth_fact_1500_test_reduced, dpa_ens_mean_fact_1500_raw_pt, dim=0)
    #print("Correlation calculated ...")
    #print(type(r_cols))
    print(type(r_cols_1300))
    print(r_cols_1300.shape)
    # mean per test member
    r_spat_mean_1300 = r_cols_1300.mean()
    if args.no_test_members > 1:
        r_spat_mean_1400 = r_cols_1400.mean()
        r_spat_mean_1500 = r_cols_1500.mean()

    r_spat_mean_all_test_members = r_spat_mean_1300
    if args.no_test_members > 1:
        r_spat_mean_all_test_members = (r_spat_mean_1300 + r_spat_mean_1400 + r_spat_mean_1500) / 3
    log_print(log_file, f"spatial mean correlation mean across test members (ETH 1300,1400,1500): {r_spat_mean_all_test_members}")

    
    
    
    # prepare for plotting
    
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    corr_spatial_pre = ut.restore_nan_columns(r_cols_1300[None, :], mask_x_te)
    print(type(corr_spatial_pre))
    print(corr_spatial_pre.shape)
    #print("Correlation spatial shape", corr_spatial_pre.shape)
    
    # turn statistics array into xarray
    # insert ds_test for coordinate information
    corr_spatial = ut.torch_to_dataarray(corr_spatial_pre, ds_test, name="Pearson correlation coefficient")
    
    # plot
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    #plot_corr = ut.plot_temperature_panel(ax, corr_spatial, corr_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    fig, ax = ut.plot_map(corr_spatial, np.linspace(0.8, 1, 9), cmap="YlOrRd", cmap_label = "Pearson Correlation")
    ax.set_title("Factual Pearson Correlation", fontsize=title_fontsize)
    plt.savefig(f"{save_path_eth}/correlation_eth_test_figure.png")

    ### Counterfactual ###
    ######################

    ## turn dpa ens (mean) into torch array
    #dpa_ens_mean_cf_1300_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1300_raw.values) #dpa_ens_mean_pt

    # calculate correlation
    #r_cols_cf = evaluation.pearsonr_cols(eth_cf_1300_test_reduced, dpa_ens_mean_cf_1300_raw_pt, dim=0)  # shape: (648,)

    ###
    ## turn dpa ens (mean) into torch array
    dpa_ens_mean_cf_1300_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1300_raw.values) #dpa_ens_mean_pt
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1400_raw.values)
        dpa_ens_mean_cf_1500_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1500_raw.values)
    # check shapes and types
    #print("dpa_ens_mean_pt shape", dpa_ens_mean_fact_1300_raw_pt.shape)

    # calculate correlation
    #print("Now calculating pearson correlation")
    # test temperature vs. eg dpa_ens_mean_1300
    r_cols_1300_cf = evaluation.pearsonr_cols(eth_cf_1300_test_reduced, dpa_ens_mean_cf_1300_raw_pt, dim=0)  # shape: (648,)
    if args.no_test_members > 1:
        r_cols_1400_cf = evaluation.pearsonr_cols(eth_cf_1400_test_reduced, dpa_ens_mean_cf_1400_raw_pt, dim=0)
        r_cols_1500_cf = evaluation.pearsonr_cols(eth_cf_1500_test_reduced, dpa_ens_mean_cf_1500_raw_pt, dim=0)
    #print("Correlation calculated ...")
    #print(type(r_cols))
    print(type(r_cols_1300))
    print(r_cols_1300.shape)
    # mean per test member
    r_spat_mean_1300_cf = r_cols_1300_cf.mean()
    if args.no_test_members > 1:
        r_spat_mean_1400_cf = r_cols_1400_cf.mean()
        r_spat_mean_1500_cf = r_cols_1500_cf.mean()

    r_spat_mean_all_test_members_cf = r_spat_mean_1300_cf
    if args.no_test_members > 1:
        r_spat_mean_all_test_members_cf = (r_spat_mean_1300_cf + r_spat_mean_1400_cf + r_spat_mean_1500_cf) / 3
    log_print(log_file, f"counterfactualspatial mean correlation mean across test set (ETH 1300,1400,1500): {r_spat_mean_all_test_members_cf}")
    ###

    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    corr_spatial_pre_cf = ut.restore_nan_columns(r_cols_1300_cf[None, :], mask_x_te)

    corr_spatial_cf = ut.torch_to_dataarray(corr_spatial_pre_cf, ds_test, name="Pearson correlation coefficient")

    # plot
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    #plot_corr = ut.plot_temperature_panel(ax, corr_spatial, corr_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    fig, ax = ut.plot_map(corr_spatial_cf, np.linspace(0.8, 1, 9), cmap="YlOrRd", cmap_label = "Pearson Correlation")
    ax.set_title("Counterfactual Pearson Correlation", fontsize=title_fontsize)
    plt.savefig(f"{save_path_eth}/cf_correlation_eth_test_figure.png")

    

    ################
    ### R2 score ###
    ################

    r2_1300 = evaluation.r2_score(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)
    if args.no_test_members > 1:
        r2_1400 = evaluation.r2_score(eth_fact_1400_test_reduced, dpa_ens_mean_fact_1400_raw_pt, dim=0)
        r2_1500 = evaluation.r2_score(eth_fact_1500_test_reduced, dpa_ens_mean_fact_1500_raw_pt, dim=0)
    
    
    print(type(r2_1300))
    print(r2_1300.shape)

    mean_r2 = r2_1300
    
    if args.no_test_members > 1:
        mean_r2 = ((r2_1300 + r2_1400 + r2_1500) / 3).mean()
    
    print("Mean R2:", mean_r2)

    
    log_print(log_file, f"Factual R^2 spatial mean, mean across test members (ETH 1300,1400,1500): {mean_r2}")



    # plot
    r2_spatial_1300_pre = ut.restore_nan_columns(r2_1300[None, :], mask_x_te)
    #print("R2 spatial", r2_spatial_pre.shape)
    
    # turn statistics array into xarray
    r2_spatial_1300 = ut.torch_to_dataarray(r2_spatial_1300_pre, ds_test, name="R^2")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_r2 = ut.plot_temperature_panel(ax, r2_spatial_1300, r2_spatial_1300.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    ax.set_title("R2", fontsize=title_fontsize)

    plt.savefig(f"{save_path_eth}/R2_eth_test_figure.png")
    
    

    

    

    #############################
    ### Reliability Index Map ###
    #############################

    ### Factual ###
    ###############
    def ri_index(dpa_fact_raw, eth_fact_test_reduced):
        # turn array into .pt array
        dpa_1300_fact_raw_pt = torch.from_numpy(dpa_fact_raw.values)
    
        
        # array to store reliability index
        ri_vals = torch.zeros((648))
    
        for n in range(648):
            # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
            ranks = ut.rank_histogram(eth_fact_test_reduced[:,n].detach().numpy(), dpa_1300_fact_raw_pt[:-1,:,n].T.detach().numpy()) # dpa_ens_pt
            counts, bin_edges = np.histogram(
                ranks, 
                bins=np.arange(ens_members + 2) - 0.5
                )
        
            # reliability index
            ri = ut.reliability_index(counts)
            ri_vals[n] = ri
        return ri_vals
        
    ri_vals_1300 = ri_index(dpa_1300_fact_raw, eth_fact_1300_test_reduced)
    
    if args.no_test_members > 1:
        ri_vals_1400 = ri_index(dpa_1400_fact_raw, eth_fact_1400_test_reduced)
        ri_vals_1500 = ri_index(dpa_1500_fact_raw, eth_fact_1500_test_reduced)

    ri_spat_mean_1300 = ri_vals_1300.mean()
    if args.no_test_members > 1:
        ri_spat_mean_1400 = ri_vals_1400.mean()
        ri_spat_mean_1500 = ri_vals_1500.mean()

    ri_spat_mean_total = ri_spat_mean_1300
    if args.no_test_members > 1:
        ri_spat_mean_total = (ri_spat_mean_1300 + ri_spat_mean_1400 + ri_spat_mean_1500) / 3
    log_print(log_file, f"Factual RI spatial mean, mean across test members (ETH 1300,1400,1500): {ri_spat_mean_total}")

    
    
    # turn array into .pt array
    #dpa_1300_fact_raw_pt = torch.from_numpy(dpa_1300_fact_raw.values)

    
    # array to store reliability index
    #ri_vals = torch.zeros((648))

    #for n in range(648):
    #    # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
    #    ranks = ut.rank_histogram(eth_fact_1300_test_reduced[:,n].detach().numpy(), dpa_1300_fact_raw_pt[:-1,:,n].T.detach().numpy()) # dpa_ens_pt
    #    counts, bin_edges = np.histogram(
    #        ranks, 
    #        bins=np.arange(ens_members + 2) - 0.5
    #        )
    
    #    # reliability index
    #    ri = ut.reliability_index(counts)
    #    ri_vals[n] = ri

    
    
    
    # PLOT
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    ri_spatial_pre = ut.restore_nan_columns(ri_vals_1300[None, :], mask_x_te)
    #print("RI spatial shape", ri_spatial_pre.shape)
    #log_print(log_file, f"Factual RI spatial mean: {ri_vals.mean()}")
    
    # turn statistics array into xarray
    ri_spatial = ut.torch_to_dataarray(ri_spatial_pre, ds_test, name="Reliability Index")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_ri = ut.plot_temperature_panel(ax, ri_spatial, ri_spatial.max().item(), levels = np.linspace(0, 1.0, 21), cbar=True)
    ax.set_title("Factual Reliability Index", fontsize = title_fontsize)
    plt.savefig(f"{save_path_eth}/factual_ri_spatial.png")

    ### Counterfactual ###
    ######################

    ri_vals_1300_cf = ri_index(dpa_1300_cf_raw, eth_cf_1300_test_reduced)
    if args.no_test_members > 1:
        ri_vals_1400_cf = ri_index(dpa_1400_cf_raw, eth_cf_1400_test_reduced)
        ri_vals_1500_cf = ri_index(dpa_1500_cf_raw, eth_cf_1500_test_reduced)

    ri_spat_mean_1300_cf = ri_vals_1300_cf.mean()
    if args.no_test_members > 1:
        ri_spat_mean_1400_cf = ri_vals_1400_cf.mean()
        ri_spat_mean_1500_cf = ri_vals_1500_cf.mean()

    ri_spat_mean_total_cf = ri_spat_mean_1300_cf
    if args.no_test_members > 1:
        ri_spat_mean_total_cf = (ri_spat_mean_1300_cf + ri_spat_mean_1400_cf + ri_spat_mean_1500_cf) / 3
    log_print(log_file, f"Counterfactual RI spatial mean, mean across test members (ETH 1300,1400,1500): {ri_spat_mean_total_cf}")
    

    

    ###---###

    # turn array into .pt array
    #dpa_1300_cf_raw_pt = torch.from_numpy(dpa_1300_cf_raw.values)

    # array to store reliability index
    #ri_vals_cf = torch.zeros((648))

    #for n in range(648):
    #    # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
    #    ranks = ut.rank_histogram(eth_cf_1300_test_reduced[:,n].detach().numpy(), dpa_1300_cf_raw_pt[:-1,:,n].T.detach().numpy()) # dpa_ens_pt
    #    counts, bin_edges = np.histogram(
    #        ranks, 
    #        bins=np.arange(ens_members + 2) - 0.5
    #        )
    
    #    # reliability index
    #    ri_cf = ut.reliability_index(counts)
    #    ri_vals_cf[n] = ri_cf

    # PLOT
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    ri_spatial_cf_pre = ut.restore_nan_columns(ri_vals_1300_cf[None, :], mask_x_te)
    
    print("RI spatial shape", ri_spatial_cf_pre.shape)
    #log_print(log_file, f"Counterfactual RI spatial mean: {ri_vals_cf.mean()}")
    
    # turn statistics array into xarray
    ri_spatial_cf = ut.torch_to_dataarray(ri_spatial_cf_pre, ds_test, name="Reliability Index")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_ri = ut.plot_temperature_panel(ax, ri_spatial_cf, ri_spatial_cf.max().item(), levels = np.linspace(0, 0.3, 13), cbar=True)
    ax.set_title("Counterfactual Reliability Index", fontsize = title_fontsize)
    plt.savefig(f"{save_path_eth}/cf_ri_spatial.png")
    
    


    ###################
    ### Time Series ###
    ###################

    # DPA ENS statistics
    # mean of DPA ensemble 
    # dpa_ens_mean

    # Standard deviation along dim=0
    std0_dpa_ens = dpa_1300_fact_raw.std(dim="ensemble_member")

    ### PLOT TIME SERIES OF 1 GRID CELL ###
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)
    #plt.figure(figsize=(12, 6))   # wider (12 inches) and not too tall (4 inches)
    ax.plot(range(100), eth_fact_1300_test_reduced[:100, 0], label="true")
    ax.plot(range(100), dpa_ens_mean_fact_1300_raw.values[:100, 0], label="restored")
    ax.fill_between(range(100), dpa_ens_mean_fact_1300_raw.values[:100, 0] - 2*std0_dpa_ens.values[:100, 0], 
                     dpa_ens_mean_fact_1300_raw.values[:100, 0] + 2*std0_dpa_ens.values[:100, 0],
                     color="tab:orange", alpha=0.2, label="±2 std")
    
    ax.legend()
    fig.tight_layout()
    ax.set_title("Standard deviation", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/time_series_dpa_mean_vs_test_member.png")
    plt.show()

    ##################################
    ### Regional mean temperatures ###
    ##################################

    
    
    ### Germany ###
    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15

    # Spain
    sp_lat_min = 38
    sp_lat_max = 42
    sp_lon_min = -8
    sp_lon_max = 0
    

   

    ####################
    ### Violin plots ###
    ####################

    ### max temperatures ###
    # fig = evaluation.plot_violins(truth = ds_test_1300_eth_fact, 
    #                         dpa_ensemble = dpa_1300_fact_restored,
    #                         save_path = f"{save_path_eth}",
    #                         lat_min = ger_lat_min,
    #                         lat_max = ger_lat_max,
    #                         lon_min = ger_lon_min,
    #                         lon_max = ger_lon_max
    #                         )



    # fig = evaluation.plot_violins(truth = ds_test_1300_eth_cf, 
    #                         dpa_ensemble = dpa_1300_cf_restored,
    #                         save_path = f"{save_path_eth}",
    #                         lat_min = ger_lat_min,
    #                         lat_max = ger_lat_max,
    #                         lon_min = ger_lon_min,
    #                         lon_max = ger_lon_max,
    #                         in_fact_for_cf = ds_test_1300_eth_fact
    #                         )


    # ### random temperatures ###
    # fig = evaluation.plot_violins(truth = ds_test_1300_eth_fact, 
    #                         dpa_ensemble = dpa_1300_fact_restored,
    #                         save_path = f"{save_path_eth}",
    #                         lat_min = ger_lat_min,
    #                         lat_max = ger_lat_max,
    #                         lon_min = ger_lon_min,
    #                         lon_max = ger_lon_max,
    #                         mode="random"
    #                         )



    # fig = evaluation.plot_violins(truth = ds_test_1300_eth_cf, 
    #                         dpa_ensemble = dpa_1300_cf_restored,
    #                         save_path = f"{save_path_eth}",
    #                         lat_min = ger_lat_min,
    #                         lat_max = ger_lat_max,
    #                         lon_min = ger_lon_min,
    #                         lon_max = ger_lon_max,
    #                         in_fact_for_cf = ds_test_1300_eth_fact,
    #                         mode="random"
    #                         )

    ######################
    ### Rank Histogram ###
    ######################

    # Factual Rank Hists
    
    ### Germany ###
    
    fig, ax = evaluation.create_rank_hist(ds_test_1300_eth_fact, 
                                             dpa_1300_fact_restored,
                                             ger_lat_min,
                                             ger_lat_max,
                                             ger_lon_min,
                                             ger_lon_max,
                                             figsize_hist,
                                             "Factual Rank histogram Germany",
                                             title_fontsize
                                             )
    
    fig.savefig(f"{save_path_eth}/Ger_rank_hist.png")

    ### Spain ###
    
    fig, ax = evaluation.create_rank_hist(ds_test_1300_eth_fact, 
                                             dpa_1300_fact_restored,
                                             sp_lat_min,
                                             sp_lat_max,
                                             sp_lon_min,
                                             sp_lon_max,
                                             figsize_hist,
                                             "Factual Rank histogram Spain",
                                             title_fontsize
                                             )
    
    fig.savefig(f"{save_path_eth}/Sp_rank_hist.png")

    # Counterfactual Rank Hists

    ### Germany ###
    
    fig, ax = evaluation.create_rank_hist(ds_test_1300_eth_cf, 
                                             dpa_1300_cf_restored,
                                             ger_lat_min,
                                             ger_lat_max,
                                             ger_lon_min,
                                             ger_lon_max,
                                             figsize_hist,
                                             "Counterfactual Rank histogram Germany",
                                             title_fontsize
                                             )
    
    fig.savefig(f"{save_path_eth}/Ger_cf_rank_hist.png")

    ### Spain ###
    
    fig, ax = evaluation.create_rank_hist(ds_test_1300_eth_cf, 
                                             dpa_1300_cf_restored,
                                             sp_lat_min,
                                             sp_lat_max,
                                             sp_lon_min,
                                             sp_lon_max,
                                             figsize_hist,
                                             "Counterfactual Rank histogram Spain",
                                             title_fontsize
                                             )
    
    fig.savefig(f"{save_path_eth}/Sp_cf_rank_hist.png")

    

    # temp_true_ger
    #temp_true_ger_pre = ds_test_1300_eth_fact.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    #cf_temp_true_ger_pre = ds_test_1300_eth_cf.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))

    
    # create weights
    # 1) define weights as above
    #weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    #w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    #temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    #cf_temp_true_ger = cf_temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    #print("cf_temp_true_ger:", cf_temp_true_ger)

    

    ### DPA Ensemble ###
    # standard deviation, germany mean
    #dpa_ens_std = dpa_1300_fact_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    #dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    #dpa_ens_mean_ger = dpa_ens_mean_fact_1300_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored
    #print(dpa_ens_mean_ger.shape)
    #dpa_ens_mean_ger_cf = dpa_ens_mean_cf_1300_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble germany average (ensemble_member, )
    #dpa_ens_ger_1300 = dpa_1300_fact_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    #print("dpa_ens_ger_1300:", dpa_ens_ger_1300.values.T.shape)
    
    
    # 2) compute ranks
    """
    truth : array_like, shape (n_cases,)
        The observed “true” values.
    ensemble : array_like, shape (n_cases, n_members)
        The ensemble forecasts for each case.
    """
    #truth = temp_true_ger.values
    #ensemble = dpa_ens_ger_1300.values.T

    #print("truth shape:", truth.shape)
    #print("ensemble shape:", ensemble.shape)
    
    #ranks = ut.rank_histogram(truth, ensemble)
    #n_members = ensemble.shape[1]
    
    # 3) plot rank histogram
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map)
    
    # bins from 0..n_members inclusive
    #ax.hist(ranks, bins=np.arange(n_members+2) - 0.5, edgecolor="black")
    
    # set x-ticks to align with bin centers
    #ax.set_xticks(np.arange(n_members + 1))
    
    # labels and title
    #ax.set_xlabel("Rank of truth among ensemble members")
    #ax.set_ylabel("Count")
    #ax.set_title("Germany Average", fontsize=title_fontsize)
    
    #plt.tight_layout()
    #fig.savefig("ETH_analysis_results/ger_rank_histogram.png")
    #plt.show()

    # end script
    

    ################################
    ### RUN LE TRAIN DATA SCRIPT ###
    ################################
    if args.include_train_analysis:
        analyse_dpa_ensemble_from_LE_train_set.main(args) 

    ###########################
    ### CREATE SUMMARY PAGE ###
    ###########################
    
    create_summary_page.summary(path_eth=save_path_eth, path_le=save_path_le, save_path=save_path_eth, period=f"{time_period[0]}-{time_period[1]}", include_train_analysis=args.include_train_analysis)
    



if __name__ == "__main__":
    main()
    
