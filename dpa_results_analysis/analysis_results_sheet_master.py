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
    ens_members=100
    ################
    ### Analysis ###
    ################
    # create figures page
    #fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # 2x2 grid of subplots

    # load test data
    print("Loading test data ...")
    z500_test, mask_x_te, ds, ds_test, x_te_reduced = de.load_test_data()
    print("x_te_reduced shape:", x_te_reduced.shape)

    
    # load DPA ensemble
    
    # including nan's
    #dpa_ensemble = xr.open_zarr(f"{save_path_ensemble_single}/dpa_ens_100_dataset_restored.zarr", consolidated=True)
    
    # RAW ensemble without NaNs
    # shape: ensemble_member: 100time: 64000lat_x_lon: 648
    print("Loading DPA ensemble ...")
    dpa_ensemble_raw = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/raw_dpa_ens_100_dataset_restored.nc")

    dpa_ensemble_restored = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/dpa_ens_100_dataset_restored.nc")
    dpa_ens_mean_restored = dpa_ensemble_restored.TREFHT.mean(dim="ensemble_member")
    
    ###########################
    ### PEARSON CORRELATION ###
    ###########################
    # prepare arrays
    print("Calculating ensemble mean ...")
    dpa_ens_mean = dpa_ensemble_raw.TREFHT.mean(dim="ensemble_member")

    ## turn dpa ens (mean) into torch array
    dpa_ens_pt = torch.from_numpy(dpa_ensemble_raw.TREFHT.values)
    print("DPA ensemble shape:", dpa_ens_pt.shape)
    dpa_ens_mean_pt = torch.from_numpy(dpa_ens_mean.values)
    
    # check shapes and types
    print("dpa_ens_mean_pt shape", dpa_ens_mean_pt.shape)
    print("DPA ens mean type:", type(dpa_ens_mean))

    # calculate correlation
    print("Now calculating pearson correlation")
    r_cols = evaluation.pearsonr_cols(x_te_reduced, dpa_ens_mean_pt, dim=0)  # shape: (648,)

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
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    #plot_corr = ut.plot_temperature_panel(ax, corr_spatial, corr_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    #plt.savefig("correlation_test_figure.pdf")

    ################
    ### R2 score ###
    ################

    r2 = evaluation.r2_score(x_te_reduced, dpa_ens_mean_pt, dim=0)

    # plot
    r2_spatial_pre = ut.restore_nan_columns(r2[None, :], mask_x_te)
    print("R2 spatial", r2_spatial_pre.shape)
    
    # turn statistics array into xarray
    r2_spatial = ut.torch_to_dataarray(r2_spatial_pre, ds_test, name="R^2")
    
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    #plot_r2 = ut.plot_temperature_panel(ax, r2_spatial, r2_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    #plt.savefig("R2_test_figure.pdf")

    #######################
    ### Rank histograms ###
    #######################

    # array to store reliability index
    ri_vals = torch.zeros((648))

    for n in range(648):
        # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
        ranks = ut.rank_histogram(x_te_reduced[:,n].detach().numpy(), dpa_ens_pt[:-1,:,n].T.detach().numpy())
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
    
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    #plot_ri = ut.plot_temperature_panel(ax, ri_spatial, ri_spatial.max().item(), levels = np.linspace(0, 0.2, 9), cbar=True)
    #plt.savefig("ri_spatial.pdf")


    ###################
    ### Time Series ###
    ###################

    # DPA ENS statistics
    # mean of DPA ensemble 
    # dpa_ens_mean

    # Standard deviation along dim=0
    std0_dpa_ens = dpa_ensemble_raw.TREFHT.std(dim="ensemble_member")

    ### PLOT TIME SERIES OF 1 GRID CELL ###
    #plt.figure(figsize=(12, 6))   # wider (12 inches) and not too tall (4 inches)
    #plt.plot(range(100), x_te_reduced[:100, 0], label="true")
    #plt.plot(range(100), dpa_ens_mean.values[:100, 0], label="restored")
    #plt.fill_between(range(100), dpa_ens_mean.values[:100, 0] - 1*std0_dpa_ens.values[:100, 0], 
                     #dpa_ens_mean.values[:100, 0] + 1*std0_dpa_ens.values[:100, 0],
                     #color="tab:orange", alpha=0.2, label="±1 std")
    
    #plt.legend()
    #plt.tight_layout()
    #plt.title("Standard deviation")
    #plt.savefig("time_series_dpa_mean_vs_test_member.pdf")
    #plt.show()


    ### Regional mean temperatures ###

    # Germany

    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15

    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = ds_test.TREFHT.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    ### DPA Ensemble ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_ensemble_restored.TREFHT.std(dim="ensemble_member")
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger = dpa_ensemble_restored.TREFHT.mean(dim="ensemble_member").sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    print(dpa_ens_mean_ger.shape)

    # plot
    n_stds = 2
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger
    
    #plt.figure(figsize=(12, 6))
    
    #temp_true_ger.isel(time=slice(103,122)).plot(label="True")
    #dpa_ens_mean_ger.isel(time=slice(103,122)).plot(label="DPA mean")
    
    
    # fill_between needs numpy arrays + axis
    #plt.fill_between(
        #dpa_ens_mean_ger.isel(time=slice(103,122)).time.values,   # x-axis values (datetime64)
        #lower_env.isel(time=slice(103,122)).values,        # lower bound
        #upper_env.isel(time=slice(103,122)).values,        # upper bound
        #color="tab:orange", alpha=0.2, label=f"+/- {n_stds} std"
    #)
    
    #plt.legend()
    #plt.tight_layout()
    #plt.title("DPA mean-Temperature in Germany Domain")
    #plt.savefig("Germany_mean_T_ts.pdf")
    #plt.show()

    # ADD COUNTERFACTUAL TEMPERATURE HERE


    ############
    ### BIAS ###
    ############

    ### Europe domain spatial average ###

    
    # reshape mask
    mask_xr = ut.torch_to_dataarray(mask_x_te.unsqueeze(0), ds, lat_dim=32, lon_dim=32, name="Mask") # mask_x_te.unsqueeze(0) adds dimension for torch_to_dataarray function to work properly


    ### DPA Ensemble ###
    dpa_ens_mean_entire_domain = dpa_ensemble_raw.isel(time=slice (-4769, 64000)).TREFHT.mean(dim = ("ensemble_member", "lat_x_lon"), skipna=True)
    print(dpa_ens_mean_entire_domain)

    plot_dpa_ens_mean_entire_domain = dpa_ens_mean_entire_domain.plot()
    plt.savefig("dpa_ens_GMT.pdf")
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
    plot_le_forced_response = le_T_mean_spat_mean.plot()
    plt.savefig("LE_Europe_Forced_response.pdf")
    plt.show()






if __name__ == "__main__":
    main()
    
