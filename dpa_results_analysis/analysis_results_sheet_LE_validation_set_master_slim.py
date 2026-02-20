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
    
    
    #####################
    ### Prepare paths ###
    #####################
    
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



    ######################
    ### Set up logfile ###
    ######################
    
    log_file = f"{save_path_eth}/test_log_metrics_{time_period[0]}-{time_period[1]}.txt"
    
    # Get current time and print it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(log_file, f"=== Current Time: {current_time} ===")
    log_print(log_file, f"=== Quantiles ===")
    #log_print(log_file, f"\n")

    
    
    
    #########################
    ### Plotting settings ###
    #########################
 
    # plotting settings
    title_fontsize = 18
    figsize_map = (10,8)
    figsize_ts = (10,8)
    figsize_hist = (8,6)



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
    
    # Large Ensemble Data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced, pi_period_mean = de.load_test_data(args.settings_file_path)
    print("x_te_reduced shape:", x_te_reduced.shape)
    print("x_te_reduced type:", type(x_te_reduced))

    # ETH Ensemble Test data
    #z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf, _, _ = de.load_eth_test_data(args.settings_file_path)
    # z500                  -> test predictors
    # mask_x_te_eth_fact    -> land mask
    # ds_test_eth_fact      -> factual test temperatures (xarray dataset) lat: 32, lon: 32, time: 14307
    # x_te_reduced_eth_fact -> land grid cells factual temperature data
    # x_te_reduced_eth_cf   -> land grid cells counterfactual temperature data

    
    slice_end_index = int(x_te_reduced.shape[0]/args.no_test_members)
    print("Slice end index:", slice_end_index)

    # split into its members


    chunks = torch.tensor_split(x_te_reduced, 10, dim=0)

    #print(chunks)
    print(chunks[0].shape)
    print(chunks[9].shape)
    
    n = ds_test.sizes["time"]
    step = n // args.no_test_members
    print("n:", n)
    print("step:", step)
    # datasets
    datasets_truth_test = [
            ds_test.TREFHT.isel(time=slice(i * step, (i + 1) * step))
            for i in range(args.no_test_members)
        ]


    
    ds_test_01 = ds_test.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1])) # HERE TP

    # get indices of time slices
    time_index = ds_test.TREFHT.isel(time=slice(0, slice_end_index)).get_index("time")
    print("Time index:", time_index)
    indices = time_index.get_indexer(ds_test.time.values)
    start_idx, end_idx = indices[0], indices[-1]+1 # add 1 to include last index

    print(ds_test_01)
    print(start_idx, end_idx)
    
    #############################

    print("Start index:", start_idx)
    print("End index:", end_idx)

    
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
    #print(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
    #print(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    #print(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
    #print(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    
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
        #dpa_ensemble_raw_cf = xr.open_dataset(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
        #dpa_ensemble_restored_cf = xr.open_dataset(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    print("DAE dataset:", dpa_ensemble_fact_raw)
    print("DAE dataset restored:", dpa_ensemble_fact_restored)
    

    # split DAE ensemble into single members
    # FACTUAL
    #dpa_01_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        #n = dpa_ensemble_fact_raw.sizes["time"]
        #step = n // args.no_test_members
        
        print("Step:", step)
        
        datasets_raw = [
            dpa_ensemble_fact_raw.TREFHT.isel(time=slice(i * step, (i + 1) * step))
            for i in range(10)
        ]

    print("Datsets raw:", datasets_raw)
    print("Example dataset:", datasets_raw[0])
    

    # shape: ensemble_member: 100, time: 14307, lat: 32, lon: 32
    #dpa_01_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        n = dpa_ensemble_fact_restored.sizes["time"]
        
        
        print("Step:", step)
        
        datasets_restored = [
            dpa_ensemble_fact_restored.TREFHT.isel(time=slice(i * step, (i + 1) * step))
            for i in range(10)
        ]

    print("Datsets restored:", datasets_restored)
    print("Example dataset:", datasets_restored[0])

    # turn both lists into lists of torch arrays
    datasets_raw_torch = [
        torch.from_numpy(x.values)
        for x in datasets_raw
    ]

    datasets_restored_torch = [
        torch.from_numpy(x.values)
        for x in datasets_restored
    ]



    
    
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
    #dpa_01_fact_raw_pt = torch.from_numpy(datasets_raw[0].TREFHT.values)
    #qu_gc_diff01, mae_spatial_mean01 = c_quantiles(chunks[0], dpa_01_fact_raw_pt, quantiles)
    #qu_gc_diff01_099, mae_spatial_mean01_099 = c_quantiles(chunks[0], dpa_01_fact_raw_pt, torch.tensor(0.99))
    #test_all_quantiles.append(mae_spatial_mean01)
    #test_099_quantile.append(mae_spatial_mean01_099)
    #print("Spatial mean of MAE per grid-cell in 5% quantiles:", mae_spatial_mean01)
    #print("Spatial mean of MAE per grid-cell in 99% quantiles:", mae_spatial_mean01_099)
    if True:
        for i,test_data in enumerate(chunks):
            print("i:",i)
            dpa_n_fact_raw_pt = torch.from_numpy(datasets_raw[i].values)
            qu_gc_diff01, mae_spatial_mean01 = c_quantiles(test_data, dpa_n_fact_raw_pt, quantiles)
            qu_gc_diff01_099, mae_spatial_mean01_099 = c_quantiles(test_data, dpa_n_fact_raw_pt, torch.tensor(0.99))
            test_all_quantiles.append(mae_spatial_mean01)
            test_099_quantile.append(mae_spatial_mean01_099)
    
        # compute mean values
        quantile_099_mean = np.mean(test_099_quantile)
        all_quantiles_mean = np.mean(test_all_quantiles)
        print("099 quantile mean MAE across test members:", quantile_099_mean)
        print("all quantiles mean MAE across test members:", all_quantiles_mean)
    
        log_print(log_file, f"all quantiles mean MAE across test members (ETH 1300,1400,1500): {all_quantiles_mean}")
        log_print(log_file, f"099 quantile mean MAE across test members (ETH 1300,1400,1500): {quantile_099_mean}")

    #############################
    ### Conditional coverages ###
    #############################
    
    # y_test_np: test truth grid cell time data
    # quantile_predictions_dpa: shape: (14307, 100), np.quantile(trefht_dpa_trans_ger_mean.values.T, np.linspace(0.05, 0.95, 19), axis=1).T
    # quantiles: 
    # dpa_1300_fact_raw: shape (100, 4769, 648)
    # trefht_dpa_trans_ger_mean.values.T.shape: (14307, 100)

    if True:

        quantiles_cq = torch.linspace(0.05, 0.95, 19)
        quantiles_cq_np = np.linspace(0.05, 0.95, 19)
    
    
        ### Factual ###
        mae_means = []
        mae099_means = []
    
        #through = zip([eth_fact_1300_test_reduced], [dpa_1300_fact_raw])
        if args.no_test_members > 1:
            #through = zip([eth_fact_1300_test_reduced, eth_fact_1400_test_reduced, eth_fact_1500_test_reduced], [dpa_1300_fact_raw, dpa_1400_fact_raw, dpa_1500_fact_raw])
            through = zip(chunks,datasets_raw_torch)
        print("Now computing conditional quantiles")
        for y_test_np, dpa_xxxx_fact_raw in through:
            mae_list = []
            mae099_list = []
            for i in range(648):
                print(i)
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


    

    ########################
    ### Autocorrelations ###
    ########################
    # deleted

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
                                      ds_coords = ds_test,
                                      ens_members=ens_members)
    
    print("DPA tensor list factual length:", len(dpa_list))
    print("DPA tensor list factual element shape:", dpa_list[0].shape) # list elements contain all timesteps 14307 (3 x 4769)

    
    start_idx = 0
    end_idx = 4769

    e_loss_list = []
    s1_loss_list = []
    s2_loss_list = []
    e_loss_field = torch.zeros(10, 648)
    s1_loss_field = torch.zeros(10, 648)
    s2_loss_field = torch.zeros(10, 648)
    for n in range(args.no_test_members):
        print("Calculating eloss", n)
        e_loss_array_n = energy_loss_per_gc(chunks[n], dpa_list, n*end_idx, (n+1)*end_idx, mask_x_te)
        e_loss_field[n,:] = e_loss_array_n[0,:]
        s1_loss_field[n,:] = e_loss_array_n[1,:]
        s2_loss_field[n,:] = e_loss_array_n[2,:]
        e_loss_test_mean = np.mean([e_loss_array_n[0,:].mean()]) # spatial means
        print("E loss test mean:", e_loss_test_mean)
        s1_loss_test_mean = np.mean([e_loss_array_n[1,:].mean()])
        s2_loss_test_mean = np.mean([e_loss_array_n[2,:].mean()])

        e_loss_list.append(e_loss_test_mean)
        s1_loss_list.append(s1_loss_test_mean)
        s2_loss_list.append(s2_loss_test_mean)

    e_loss_validation_mean = np.mean(e_loss_list) # mean across validation members
    s1_loss_validation_mean = np.mean(s1_loss_list)
    s2_loss_validation_mean = np.mean(s2_loss_list)

    print(e_loss_validation_mean)
    print(s1_loss_validation_mean)
    print(s2_loss_validation_mean)


    log_print(log_file, f"Factual Energy-Score spatial mean across test members: {e_loss_test_mean}")
    log_print(log_file, f"Factual S1-Score spatial mean across test members: {s1_loss_test_mean}")
    log_print(log_file, f"Factual S2-Score spatial mean across test members: {s2_loss_test_mean}")
    
    ### Plot some energy loss ###
    #############################

    # caluclate e loss map
    e_loss_array_01 = e_loss_field.mean(dim=0)#energy_loss_per_gc(chunks[0], dpa_list, 0, 4769, mask_x_te)
    s1_loss_array_01 = s1_loss_field.mean(dim=0)
    s2_loss_array_01 = s2_loss_field.mean(dim=0)
    print(e_loss_array_01.shape)
         
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array_01.unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(s1_loss_array_01.unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(s2_loss_array_01.unsqueeze(0), mask_x_te)

    
    # transform e_loss_array into xarray
    e_loss_xr = ut.torch_to_dataarray(x_tensor = e_loss_restored, coords_ds = ds_test, lat_dim=32, lon_dim=32, name="energy_loss_total") # add oth dimensin 
    s1_loss_xr = ut.torch_to_dataarray(x_tensor = s1_loss_restored, coords_ds = ds_test, lat_dim=32, lon_dim=32, name="s1_loss")
    s2_loss_xr = ut.torch_to_dataarray(x_tensor = s2_loss_restored, coords_ds = ds_test, lat_dim=32, lon_dim=32, name="s2_loss")
    #print("E loss xr:", e_loss_xr)
    
    #energy_levels = np.linspace(0.4, 1.1, 7)
    #levels = np.linspace(0.8, 2.0, 13)
    energy_levels = np.linspace(0.5, 1.5, 21)
    #levels = np.linspace(0.0, 3.0, 11)
    
    # plot energy score
    fig, ax = ut.plot_map(e_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Energy Loss")
    ax.set_title("Factual Energy Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/energy_loss_map.png")
    plt.show()

    # plot S1 loss
    fig, ax = ut.plot_map(s1_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Reconstruction Loss (S1)")
    ax.set_title("Factual S1 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/S1_loss_map.png")
    plt.show()

    # plot s2 loss
    fig, ax = ut.plot_map(s2_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Variability Loss (S2)")
    ax.set_title("Factual S2 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/S2_loss_map.png")
    plt.show()


    
    
    ###########################
    ### PEARSON CORRELATION ###
    ###########################
    # pearson correlation
    correlation_spat_mean_test_membs = [] # listr for saving spatial means
    r2_spat_mean_test_membs = []
    for i in range(args.no_test_members):
        # calculate ensemble mean
        dpa_ens_mean_fact = datasets_raw[i].mean(dim="ensemble_member")
        print("DPA ens mean shape:", dpa_ens_mean_fact.shape)
    
        # turn into torch array
        dpa_ens_mean_fact_raw = torch.from_numpy(dpa_ens_mean_fact.values) #dpa_ens_mean_pt
        print(dpa_ens_mean_fact_raw.shape)
        
        # calculate correlation per validation member
        print("Chunks element shape:", chunks[i].shape)
        r_cols_n = evaluation.pearsonr_cols(chunks[i], dpa_ens_mean_fact_raw, dim=0)
        print("Shape correlations:", r_cols_n.shape)
        correlation_spat_mean_test_membs.append(r_cols_n.mean())

        # R^2
        r2_n = evaluation.r2_score(chunks[i], dpa_ens_mean_fact_raw, dim=0)
        print("R2 shape:", r2_n.shape)
        r2_spat_mean_test_membs.append(r2_n.mean())
        
        ###
        # mean per test member
    
        

    r_spat_mean_all_test_members = np.mean(correlation_spat_mean_test_membs)
    r2_spat_mean_all_test_members = np.mean(r2_spat_mean_test_membs)
    print("Mean of spatial mean of local correlations:", r_spat_mean_all_test_members)
    print("Mean of spatial mean of local R2:", r2_spat_mean_all_test_members)
    log_print(log_file, f"spatial mean correlation mean across test members (ETH 1300,1400,1500): {r_spat_mean_all_test_members}")
    log_print(log_file, f"Factual R^2 spatial mean, mean across test members (ETH 1300,1400,1500): {r2_spat_mean_all_test_members}")

    

    

    #############################
    ### Reliability Index Map ###
    #############################

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

    ri_s_spat = torch.zeros(args.no_test_members, 648)
    ri_s_spat_mean = []
    for i in range(args.no_test_members):
        ri_vals_n = ri_index(datasets_raw[i], chunks[i])
        
        ri_s_spat_mean.append(ri_vals_n.mean())
        ri_s_spat[i,:]=ri_vals_n

        
    print("ri vals:", ri_s_spat_mean)
    log_print(log_file, f"Factual RI spatial mean, mean across validation members: {np.mean(ri_s_spat_mean)}")

    ########
    mean_ris = ri_s_spat.mean(dim=0)
    print("mean ris shape:", mean_ris.shape)
    
    ri_spatial_pre = ut.restore_nan_columns(mean_ris[None, :], mask_x_te)
    
    # turn statistics array into xarray
    ri_spatial = ut.torch_to_dataarray(ri_spatial_pre, ds_test, name="Reliability Index")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_ri = ut.plot_temperature_panel(ax, ri_spatial, ri_spatial.max().item(), levels = np.linspace(0, 0.5, 21), cbar=True)
    ax.set_title("Factual Reliability Index", fontsize = title_fontsize)
    plt.savefig(f"{save_path_eth}/factual_ri_spatial.png")

    #######
    # Factual Rank Hists
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
    ### Germany ###
    
    fig, ax = evaluation.create_rank_hist(datasets_truth_test[0], 
                                             datasets_restored[0],
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
    
    fig, ax = evaluation.create_rank_hist(datasets_truth_test[0], 
                                             datasets_restored[0],
                                             sp_lat_min,
                                             sp_lat_max,
                                             sp_lon_min,
                                             sp_lon_max,
                                             figsize_hist,
                                             "Factual Rank histogram Spain",
                                             title_fontsize
                                             )
    
    fig.savefig(f"{save_path_eth}/Sp_rank_hist.png")
    

    



if __name__ == "__main__":
    main()
    
