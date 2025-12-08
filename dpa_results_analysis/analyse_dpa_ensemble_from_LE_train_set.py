import torch
#from torchvision.utils import make_grid
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
#from sklearn.manifold import TSNE
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import shutil
import argparse

import sys
sys.path.append('../')
import dpa_ensemble as de
import utils as ut
import evaluation

def get_parser():
    """Return an argument parser for this module."""
    parser = argparse.ArgumentParser(description="Analyse DPA ensemble from LE train set")
    parser.add_argument("--period_start", type=int, required=True, help="Start year")
    parser.add_argument("--period_end", type=int, required=True, help="End year")
    parser.add_argument("--ens_members", type=int, default=100, help="Number of ensemble members")
    parser.add_argument("--save_path_le", type=str, required=True, help="Save path for LE analysis figures")
    parser.add_argument("--save_path_eth", type=str, required=True, help="Save path for ETH analysis figures")
    parser.add_argument("--ensemble_path", type=str, help="Path of DPA ensemble")
    parser.add_argument("--no_epochs", type=int, help="Number of epochs")
    parser.add_argument("--settings_file_path", type=str, help="Path of settings (datasets) to create ensemble.")

    return parser
    
def main(args=None):

    print("Script starting")
    #######################
    ### INPUT ARGUMENTS ###
    #######################
    #parser = argparse.ArgumentParser(description="Example script with arguments")

    # Define arguments
    #parser.add_argument("--period_start", type=int, help="Start year of period to analyse")
    #parser.add_argument("--period_end", type=int, help="End year of period to analyse")
    #parser.add_argument("--ensemble_path", type=str, default=10, help="Path of DPA ensemble")
    #parser.add_argument("--no_epochs", type=int, help="Number of epochs model was trained used for creating this DPA ensemble")
    

    #args = parser.parse_args()
    parser = get_parser()

    # if called as standalone script
    if args is None:
        args = parser.parse_args()

    print("Args parsed")
    
    save_path_le = args.save_path_le #f"ETH_analysis_results/final_analysis_train_LE/model_trained_for_{args.no_epochs}_epochs/"
    print("save path LE analysis results:", save_path_le)
    os.makedirs(save_path_le, exist_ok=True)

    # settings
    # plotting settings
    title_fontsize = 18
    figsize_map = (8,6)
    figsize_ts = (10,8)
    
    #time_period = ["1850", "2100"]
    time_period = [str(args.period_start), str(args.period_end)]

    # years for time series plotting
    years = ["1920", "1940", "1960", "1980", "2000", "2020", "2040", "2060"]#, "2000", "2020", "2040", "2060"]
    
    ens_members=100

    
    # load data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced = de.load_test_data(args.settings_file_path)
    print("x_tr_reduced:", x_tr_reduced.shape)
    print("ds_train shape:", ds_train.TREFHT.shape)
    
    ####################
    ### Energy Score ###
    ####################
    
    ### Factual ###
    ###############

    # dpa ensemble for one grid cell list[(time steps,1), (time steps,1), ...]
    # this is the reconstructed array containing 1024 grid cells (many with NaNs)
    _, dpa_list, _, _, _ = ut.load_dpa_arrays(path=f"{args.ensemble_path}/train_set_ensemble_after_{args.no_epochs}_epochs/",
                                  mask = mask_x_te,
                                  ds_coords = ds_train,
                                  ens_members=ens_members)
    print("DPA tensor list:", len(dpa_list))
    print("DPA tensor list element shape:", dpa_list[0].shape) # list elements contain all timesteps 14307 (3 x 4769)
    
    if True:
        e_loss_array = torch.zeros(3,648)
        for i in range(648):
            # keep only subset of tensors in list
            dpa_list_subset = [t[:, i] for t in dpa_list] #[t[:4769, i] for t in dpa_list]
        
            ### calculate energy score ###
            ### insert true data here!!
            #e_loss = energy_loss(eth_fact_1300_test_reduced[:,i], dpa_list_subset)
            e_loss = energy_loss(x_tr_reduced[:,i], dpa_list_subset)
            e_loss_array[0,i] = e_loss[0].detach() 
            e_loss_array[1,i] = e_loss[1].detach()
            e_loss_array[2,i] = e_loss[2].detach()
            print(i)
            print("Energy Loss:", e_loss)
    
        

        # save e_loss:
        torch.save(e_loss_array, f"{args.ensemble_path}/train_set_ensemble_after_{args.no_epochs}_epochs/train_e_loss_spatial.pt")
    else:
        # load e_loss array:
        e_loss_array = torch.load(f"{args.ensemble_path}/train_set_ensemble_after_{args.no_epochs}_epochs/train_e_loss_spatial.pt")

    print("E Loss shape:", e_loss_array.shape)
    print("type mask:", type(mask_x_te))
    print("mask shape:", mask_x_te.shape)
    print("type e loss array:", type(e_loss_array))
    
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array[0,:].unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(e_loss_array[1,:].unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(e_loss_array[2,:].unsqueeze(0), mask_x_te)

    print("e_loss_restored shape:", e_loss_restored.shape)

    
    # transform e_loss_array into xarray
    e_loss_xr = ut.torch_to_dataarray(x_tensor = e_loss_restored, coords_ds = ds_train, lat_dim=32, lon_dim=32, name="energy_loss_total") # add oth dimensin 
    s1_loss_xr = ut.torch_to_dataarray(x_tensor = s1_loss_restored, coords_ds = ds_train, lat_dim=32, lon_dim=32, name="s1_loss")
    s2_loss_xr = ut.torch_to_dataarray(x_tensor = s2_loss_restored, coords_ds = ds_train, lat_dim=32, lon_dim=32, name="s2_loss")
    print("E loss xr:", e_loss_xr)

    
    #energy_levels = np.linspace(0.4, 1.1, 7)
    #levels = np.linspace(0.8, 2.0, 13)
    energy_levels = np.linspace(0.0, 3.0, 11)
    levels = np.linspace(0.0, 3.0, 11)
    
    # plot energy score
    fig, ax = ut.plot_map(e_loss_xr, energy_levels, cmap="YlOrRd", cmap_label = "Energy Loss")
    ax.set_title("Train Set Energy Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_le}/LE_train_set_energy_loss_map.png")
    plt.show()

    # plot S1 loss
    fig, ax = ut.plot_map(s1_loss_xr, levels, cmap="YlOrRd", cmap_label = "Reconstruction Loss (S1)")
    ax.set_title("Train Set S1 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_le}/LE_train_set_S1_loss_map.png")
    plt.show()

    # plot s2 loss
    fig, ax = ut.plot_map(s2_loss_xr, levels, cmap="YlOrRd", cmap_label = "Variability Loss (S2)")
    ax.set_title("Train Set S2 Loss", fontsize=title_fontsize)
    fig.savefig(f"{save_path_le}/LE_train_set_S2_loss_map.png")
    plt.show()


    # now transpose (switch i and :) everything to calculate scores over time i.e. for each map
    
    if True:
    
        e_loss_array = torch.zeros(dpa_list[0].shape[0], 3)
        for i in range(dpa_list[0].shape[0]):
            # keep only subset of tensors in list
            dpa_list_subset = [t[i, :] for t in dpa_list] 
        
            ### calculate energy score ###
            e_loss = energy_loss(x_tr_reduced[i,:], dpa_list_subset)
            e_loss_array[i,0] = e_loss[0].detach() 
            e_loss_array[i,1] = e_loss[1].detach()
            e_loss_array[i,2] = e_loss[2].detach()
            print(i)
            print("Energy Loss:", e_loss)

        # Save
        torch.save(e_loss_array, f"{args.ensemble_path}/train_set_ensemble_after_{args.no_epochs}_epochs/train_e_loss_time_resolved.pt")

    #################
    ### Plot Loss ###
    #################

    else:
        ### time resolved e_loss ###
        e_loss_array = torch.load(f"{args.ensemble_path}/train_set_ensemble_after_{args.no_epochs}_epochs/train_e_loss_time_resolved.pt")
    #print("e loss shape:", e_loss_pre.shape)

    # create xarray from e_loss
    loss_types = ["energy_loss", "S1", "S2"]

    # Create time coordinate (example: daily timestamps)
    time_coord = ds_train.time
    
    # Convert torch → numpy and wrap into xarray
    eloss_xr = xr.DataArray(
        e_loss_array.detach().numpy(),                     # convert to NumPy
        dims=("time", "loss"),                 # name the dimensions
        coords={
            "time": time_coord,
            "loss": loss_types
        },
        name="loss_values"
    )
    
    #print(eloss_xr)

    # calculate mean across members in train set
    eloss_xr_time_series = eloss_xr.groupby("time").mean(dim="time")
    #print(eloss_xr_time_series)

    ### load AEROSOL optical depth ###
    aer_pre = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/other_data/JJA_mean_LE2_AODVIS_1850_2100.nc").AODVIS
    
    # Shift and then sort
    aodvis = aer_pre.assign_coords(lon=((aer_pre.lon + 180) % 360) - 180)
    aodvis = aodvis.sortby("lon")
    
    #print(aodvis)

    # standardize loss values
    # Standardize along the 'time' dimension
    eloss_standardized = (eloss_xr_time_series - eloss_xr_time_series.mean(dim="time")) / eloss_xr_time_series.std(dim="time")
    print(eloss_standardized)

    # compute yearly mean values for energy loss
    yearly_eloss_standardized_pre = (eloss_standardized.groupby("time.year").mean(dim="time"))
    #print("Yearly e loss standardized:", yearly_eloss_standardized_pre)
    years_no = (yearly_eloss_standardized_pre["year"])
    print("Number of different years:", years_no.shape)
    
    # Replace the coordinate correctly
    yearly_eloss_standardized = yearly_eloss_standardized_pre.assign_coords(
        year=("year", aodvis.time.values[-years_no.shape[0]:]))

    # subset aerosols to europe
    europe_aodvis = aodvis.sel(lat=slice(34, 64), lon=slice(-11.25, 27.5))
    print(europe_aodvis)
    
    # --- 3. Compute latitude-based weights ---
    weights = np.cos(np.deg2rad(europe_aodvis.lat))
    
    # --- 4. Compute the weighted spatial mean ---
    # Modern xarray method (recommended)
    europe_mean = europe_aodvis.weighted(weights).mean(dim=("lat", "lon"))
    
    # --- 6. Print summary ---
    print(europe_mean)

    # standardize AODVIS
    # --- 5. Standardize (z-score normalization) ---
    # (value - mean) / std  → across the time dimension
    
    europe_mean_std = (europe_mean - europe_mean.mean(dim="time")) / europe_mean.std(dim="time")
    
    
    ### PLOT ###

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize_ts)
    
    # Plot standardized energy loss (yearly)
    yearly_eloss_standardized.sel(loss="energy_loss").plot(
        ax=ax,
        label="Energy Loss (Standardized)",
        color="tab:blue",
        linewidth=1
    )
    
    # Plot standardized Europe-mean AODVIS
    europe_mean_std.plot(
        ax=ax,
        label="AODVIS (Standardized)",
        color="tab:orange",
        linewidth=1
    )
    
    # Add legend and labels
    ax.legend(fontsize=10)
    ax.set_title("Standardized Energy Loss vs. European AODVIS", fontsize=title_fontsize)
    ax.set_xlabel("Year", fontsize=title_fontsize)
    ax.set_ylabel("Standardized Value", fontsize=title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    fig.savefig(f"{save_path_le}/eloss_and_aerosols.png", dpi=300)
    plt.show()


    # all losses
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize_ts)
    
    # Plot all three losses
    yearly_eloss_standardized.sel(loss="energy_loss").plot(
        ax=ax,
        label="Energy Loss",
        color="tab:blue",
        linewidth=2
    )
    
    yearly_eloss_standardized.sel(loss="S1").plot(
        ax=ax,
        label="S1",
        color="tab:orange",
        linewidth=2
    )
    
    yearly_eloss_standardized.sel(loss="S2").plot(
        ax=ax,
        label="S2",
        color="tab:green",
        linewidth=2
    )
    
    # Customize labels and legend
    ax.legend(fontsize=10, loc="best")
    ax.set_title("Train Loss Components (yearly averages)", fontsize=title_fontsize)
    ax.set_xlabel("Year", fontsize=title_fontsize)
    ax.set_ylabel("Loss (Standardized)", fontsize=title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Optimize layout and save
    plt.tight_layout()
    fig.savefig(f"{save_path_le}/all_losses.png", dpi=300)
    plt.show()
        







if __name__ == "__main__":
    main()