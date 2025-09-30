import torch
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss, energy_loss_two_sample

import xarray as xr
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.manifold import TSNE
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil

import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import dpa_ensemble as de
import utils as ut
import evaluation


    


def main():
    # plotting settings
    title_fontsize = 18
    figsize_map = (8,6)
    figsize_ts = (10,6)
    
    ens_members=100
    ################
    ### Analysis ###
    ################
    # create figures page
    #fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # 2x2 grid of subplots

    # load test data
    print("Loading test data ...")
    
    # Large Ensemble Data
    z500_test, mask_x_te, ds, ds_test, x_te_reduced = de.load_test_data()
    print("x_te_reduced shape:", x_te_reduced.shape)

    # ETH Ensemble Test data
    z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf = de.load_eth_test_data()
    # z500                  -> test predictors
    # mask_x_te_eth_fact    -> land mask
    # ds_test_eth_fact      -> factual test temperatures (xarray dataset) lat: 32, lon: 32, time: 14307
    # x_te_reduced_eth_fact -> land grid cells factual temperature data
    # x_te_reduced_eth_cf   -> land grid cells counterfactual temperature data

    # datasets
    ds_test_1300_eth_fact = ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769))
    print("ds_test_1300_eth_fact:", ds_test_1300_eth_fact)
    ds_test_1300_eth_cf = ds_test_eth_cf.TREFHT.isel(time=slice(0, 4769))

    
    # pytorch arrays
    eth_fact_1300_test_reduced = x_te_reduced_eth_fact[:4769,:]
    eth_fact_1400_test_reduced = x_te_reduced_eth_fact[4769:2*4769,:]
    eth_fact_1500_test_reduced = x_te_reduced_eth_fact[-4769:14307,:]
    print("eth_fact_1300_test_reduced shape:", eth_fact_1300_test_reduced.shape)
    mask_x_te = mask_x_te_eth_fact
    
    # load DPA ensemble
    
    # including nan's
    #dpa_ensemble = xr.open_zarr(f"{save_path_ensemble_single}/dpa_ens_100_dataset_restored.zarr", consolidated=True)
    
    # RAW ensemble without NaNs
    # shape: ensemble_member: 100time: 64000lat_x_lon: 648
    print("Loading DPA ensemble ...")

    # FACTUAL 
    # shape: ensemble_member: 100, time: 14307, lat_x_lon: 648
    dpa_ensemble_fact_raw = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/eth_ensemble_after_30_epochs/raw_ETH_gen_dpa_ens_100_dataset.nc")
    dpa_1300_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, 4769))
    dpa_1400_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(4769,2*4769))
    dpa_1500_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(-4769,14307))
    print("dpa_1300_fact_raw:", dpa_1300_fact_raw)
    

    # shape: ensemble_member: 100, time: 14307, lat: 32, lon: 32
    dpa_ensemble_fact_restored = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/eth_ensemble_after_30_epochs/ETH_gen_dpa_ens_100_dataset_restored.nc")
    dpa_1300_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(0, 4769))
    dpa_1400_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(4769,2*4769))
    dpa_1500_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(-4769,14307))
    

    # COUNTERFACTUAL
    dpa_ensemble_raw_cf = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/eth_ensemble_after_30_epochs/raw_ETH_cf_gen_dpa_ens_100_dataset.nc")
    dpa_1300_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(0, 4769))
    dpa_1400_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(4769,2*4769))
    dpa_1500_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(-4769,14307))

    dpa_ensemble_restored_cf = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/eth_ensemble_after_30_epochs/ETH_cf_gen_dpa_ens_100_dataset_restored.nc")
    dpa_1300_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(0, 4769))
    dpa_1400_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(4769,2*4769))
    dpa_1500_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(-4769,14307))
    
    ###########################
    ### PEARSON CORRELATION ###
    ###########################
    
    # mean of restored factual DPA ensemble
    #dpa_ens_mean_restored = dpa_ensemble_fact_restored.TREFHT.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1300_restored = dpa_1300_fact_restored.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1400_restored = dpa_1400_fact_restored.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1500_restored = dpa_1500_fact_restored.mean(dim="ensemble_member")

    # mean of restored counterfactual DPA ensemble
    dpa_ens_mean_cf_1300_restored = dpa_1300_cf_restored.mean(dim="ensemble_member")
    dpa_ens_mean_cf_1400_restored = dpa_1400_cf_restored.mean(dim="ensemble_member")
    dpa_ens_mean_cf_1500_restored = dpa_1500_cf_restored.mean(dim="ensemble_member")

    # mean of raw factual ensemble
    dpa_ens_mean_fact_1300_raw = dpa_1300_fact_raw.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1400_raw = dpa_1400_fact_raw.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1500_raw = dpa_1500_fact_raw.mean(dim="ensemble_member")

    # mean of raw counterfactual ensemble
    dpa_ens_mean_cf_1300_raw = dpa_1300_cf_raw.mean(dim="ensemble_member")
    dpa_ens_mean_cf_1400_raw = dpa_1400_cf_raw.mean(dim="ensemble_member")
    dpa_ens_mean_cf_1500_raw = dpa_1500_cf_raw.mean(dim="ensemble_member")



    # mean of RAW factual ensemble
    print("Calculating ensemble mean ...")
    #dpa_fact_1300_raw_mean = dpa_ens_mean_fact_1300_raw.mean(dim="ensemble_member")

    ## turn dpa ens (mean) into torch array
    dpa_ens_mean_fact_1300_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1300_raw.values) #dpa_ens_mean_pt
    
    # check shapes and types
    print("dpa_ens_mean_pt shape", dpa_ens_mean_fact_1300_raw_pt.shape)

    # calculate correlation
    print("Now calculating pearson correlation")
    # test temperature vs. eg dpa_ens_mean_1300
    r_cols = evaluation.pearsonr_cols(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)  # shape: (648,)

    print("Correlation calculated ...")
    print(type(r_cols))
    
    # prepare for plotting
    
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    corr_spatial_pre = ut.restore_nan_columns(r_cols[None, :], mask_x_te)
    print("Correlation spatial shape", corr_spatial_pre.shape)
    
    # turn statistics array into xarray
    # insert ds_test for coordinate information
    corr_spatial = ut.torch_to_dataarray(corr_spatial_pre, ds_test, name="Pearson correlation coefficient")
    
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_corr = ut.plot_temperature_panel(ax, corr_spatial, corr_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    ax.set_title("Pearson Correlation", fontsize=title_fontsize)
    plt.savefig("ETH_analysis_results/correlation_test_figure.png")

    ################
    ### R2 score ###
    ################

    r2 = evaluation.r2_score(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)

    # plot
    r2_spatial_pre = ut.restore_nan_columns(r2[None, :], mask_x_te)
    print("R2 spatial", r2_spatial_pre.shape)
    
    # turn statistics array into xarray
    r2_spatial = ut.torch_to_dataarray(r2_spatial_pre, ds_test, name="R^2")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_r2 = ut.plot_temperature_panel(ax, r2_spatial, r2_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    ax.set_title("R2", fontsize=title_fontsize)

    plt.savefig("ETH_analysis_results/R2_test_figure.png")

    #######################
    ### Rank histograms ###
    #######################

    # turn array into .pt array
    dpa_1300_fact_raw_pt = torch.from_numpy(dpa_1300_fact_raw.values)


    
    # array to store reliability index
    ri_vals = torch.zeros((648))

    for n in range(648):
        # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
        ranks = ut.rank_histogram(eth_fact_1300_test_reduced[:,n].detach().numpy(), dpa_1300_fact_raw_pt[:-1,:,n].T.detach().numpy()) # dpa_ens_pt
        counts, bin_edges = np.histogram(
            ranks, 
            bins=np.arange(ens_members + 2) - 0.5
            )
    
        # reliability index
        ri = ut.reliability_index(counts)
        ri_vals[n] = ri

    # PLOT
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    ri_spatial_pre = ut.restore_nan_columns(ri_vals[None, :], mask_x_te)
    print("RI spatial shape", ri_spatial_pre.shape)
    
    # turn statistics array into xarray
    ri_spatial = ut.torch_to_dataarray(ri_spatial_pre, ds_test, name="Reliability Index")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_ri = ut.plot_temperature_panel(ax, ri_spatial, ri_spatial.max().item(), levels = np.linspace(0, 0.3, 13), cbar=True)
    ax.set_title("Reliability Index", fontsize = title_fontsize)
    plt.savefig("ETH_analysis_results/ri_spatial.png")

    ### PLOT SOME RANK HISTOGRAMS ###


    ###################
    ### Time Series ###
    ###################

    # DPA ENS statistics
    # mean of DPA ensemble 
    # dpa_ens_mean

    # Standard deviation along dim=0
    std0_dpa_ens = dpa_1300_fact_raw.std(dim="ensemble_member")
    # std0_dpa_ens = dpa_ensemble_raw.TREFHT.std(dim="ensemble_member") # --> original code

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
    fig.savefig("ETH_analysis_results/time_series_dpa_mean_vs_test_member.png")
    plt.show()


    ### Regional mean temperatures ###

    # Germany

    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15

    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = ds_test_1300_eth_fact.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    cf_temp_true_ger_pre = ds_test_1300_eth_cf.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))

    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    cf_temp_true_ger = cf_temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    print("cf_temp_true_ger:", cf_temp_true_ger)


    ### DPA Ensemble ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_1300_fact_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger = dpa_ens_mean_fact_1300_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored
    print(dpa_ens_mean_ger.shape)
    dpa_ens_mean_ger_cf = dpa_ens_mean_cf_1300_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300 = dpa_1300_fact_restored.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    print("dpa_ens_ger_1300:", dpa_ens_ger_1300.values.T.shape)
    
    
    # plot
    n_stds = 2
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger

    print("dpa_ens_mean_ger:", dpa_ens_mean_ger)
    print("lower env:", lower_env)
    print("upper env:", upper_env)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    year = "2040"
    temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="Factual Truth")
    dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA factual mean")
    
    # ADD COUNTERFACTUAL TEMPERATURE HERE
    cf_temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="CF Truth")
    dpa_ens_mean_ger_cf.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA CF mean")
    # ################################## #
    
    
    # fill_between needs numpy arrays + axis
    ax.fill_between(
        dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values,   # x-axis values (datetime64)
        lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
        upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
        color="tab:orange", alpha=0.2, label=f"+/- {n_stds} std"
    )

    
    
    ax.legend()
    #plt.tight_layout()
    ax.set_title("DPA mean-Temperature in Germany Domain", fontsize=title_fontsize)
    fig.savefig("ETH_analysis_results/Germany_mean_T_ts.png")
    plt.show()

    ### Germany Rank Histogram ###
    # 2) compute ranks
    """
    truth : array_like, shape (n_cases,)
        The observed “true” values.
    ensemble : array_like, shape (n_cases, n_members)
        The ensemble forecasts for each case.
    """
    truth = temp_true_ger.values
    ensemble = dpa_ens_ger_1300.values.T

    print("truth shape:", truth.shape)
    print("ensemble shape:", ensemble.shape)
    
    ranks = ut.rank_histogram(truth, ensemble)
    n_members = ensemble.shape[1]
    
    # 3) plot rank histogram
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map)
    
    # bins from 0..n_members inclusive
    ax.hist(ranks, bins=np.arange(n_members+2) - 0.5, edgecolor="black")
    
    # set x-ticks to align with bin centers
    ax.set_xticks(np.arange(n_members + 1))
    
    # labels and title
    ax.set_xlabel("Rank of truth among ensemble members")
    ax.set_ylabel("Count")
    ax.set_title("Germany Average", fontsize=title_fontsize)
    
    plt.tight_layout()
    fig.savefig("ETH_analysis_results/ger_rank_histogram.png")
    plt.show()

    


    ############
    ### BIAS ###
    ############

    ### Europe domain spatial average ###

    
    # reshape mask
    mask_xr = ut.torch_to_dataarray(mask_x_te.unsqueeze(0), ds, lat_dim=32, lon_dim=32, name="Mask") # mask_x_te.unsqueeze(0) adds dimension for torch_to_dataarray function to work properly


    ### DPA Ensemble ###
    dpa_ens_mean_entire_domain = dpa_1300_fact_raw.mean(dim = ("ensemble_member", "lat_x_lon"), skipna=True) #dpa_ensemble_raw
    print(dpa_ens_mean_entire_domain)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    dpa_ens_mean_entire_domain.plot(ax=ax)
    #fig.savefig("ETH_analysis_results/dpa_ens_GMT.png")
    plt.show()

    
    ### LE Forced response Europe ###
    # mask large ensemble    
    le_masked = ds.TREFHT.where(mask_xr.isel(time=0))

    # 'unstack' LE
    arr = le_masked   # shape: (time, lat, lon)
    
    n_ens = 100
    n_time = arr.sizes["time"] // n_ens
    
    reshaped = arr.values.reshape(arr.sizes["lat"], arr.sizes["lon"], n_ens, n_time)
    print(reshaped.shape) # shape = (32, 32, 100, 4769)
    arr_unstacked = xr.DataArray(
        reshaped,
        dims=("lat", "lon", "ensemble_member", "time"),
        coords={
            "ensemble_member": np.arange(1, n_ens + 1),
            "time": ds.time.isel(time=slice(0,4769)),
            "lat": arr.lat,
            "lon": arr.lon,
        },
        name="TREFHT"
    )
    print(arr_unstacked)
    le_T_mean_spat_mean = arr_unstacked.mean(dim=("ensemble_member", "lat", "lon"), skipna=True)

    ### compute yearly means ###
    yearly_mean_dpa = dpa_ens_mean_entire_domain.groupby("time.year").mean()
    yearly_mean_le = le_T_mean_spat_mean.groupby("time.year").mean()


    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)
    #dpa_ens_mean_entire_domain.plot(ax=ax, label="Europe DPA Ensemble Mean")
    #le_T_mean_spat_mean.plot(ax=ax, label="Europe LE Forced Response")

    yearly_mean_dpa.plot(ax=ax, label="Europe DPA Ensemble Mean")
    yearly_mean_le.plot(ax=ax, label="Europe LE Forced Response")

    print("dpa_ens_mean_entire_domain:", dpa_ens_mean_entire_domain)
    print("le_T_mean_spat_mean:", le_T_mean_spat_mean)
    
    ax.legend()
    ax.set_title("European Domain Average Temperature", fontsize=title_fontsize)
    fig.savefig("ETH_analysis_results/LE_Europe_Forced_response.png")
    plt.show()






if __name__ == "__main__":
    main()
    
