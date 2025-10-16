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
    
    parser.add_argument("--include_train_analysis", type=int, default=1, help="Whether to include analysis of train data.")
    parser.add_argument("--period_start", type=int, help="Start year of period to analyse")
    parser.add_argument("--period_end", type=int, help="End year of period to analyse")
    parser.add_argument("--ensemble_path", type=str, help="Path of DPA ensemble")
    parser.add_argument("--no_epochs", type=int, help="Number of epochs model was trained used for creating this DPA ensemble")
    parser.add_argument("--ens_members", type=int, default=100, help="Number of members in DPA ensemble")
    parser.add_argument("--save_path_le", type=str, help="Save path of LE train set analysis figures")
    parser.add_argument("--save_path_eth", type=str, help="Save path of ETH set analysis figures")

    args = parser.parse_args()
    #print(type(args))
    
    #time_period = ["2000", "2050"]
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
    
    #create_summary_page.summary(path_eth=save_path_eth, path_le=save_path_le, save_path=save_path_eth, period=f"{time_period[0]}-{time_period[1]}", include_train_analysis=0)
    
    os.makedirs(save_path_eth, exist_ok=True)
    os.makedirs(save_path_le, exist_ok=True)

    
    log_file = f"{save_path_eth}/log_metrics_{time_period[0]}-{time_period[1]}.txt"
    # Get current time and print it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(log_file, f"=== Current Time: {current_time} ===")


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
    years = ["1920", "1940", "1960", "1980", "2000", "2020", "2040", "2060"]#, "2000", "2020", "2040", "2060"]
    # subset to years that are contained in time period!!
    # Convert to integers
    start, end = map(int, time_period)
    
    # Filter years that fall within the range
    years = [y for y in map(int, years) if start <= y <= end]
    
    #print("years contained", years)
    
    ens_members=args.ens_members
    ################
    ### Analysis ###
    ################
    # create figures page
    #fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # 2x2 grid of subplots

    # load test data
    #print("Loading test data ...")
    
    # Large Ensemble Data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced = de.load_test_data()
    #print("x_te_reduced shape:", x_te_reduced.shape)

    # ETH Ensemble Test data
    z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf = de.load_eth_test_data()
    # z500                  -> test predictors
    # mask_x_te_eth_fact    -> land mask
    # ds_test_eth_fact      -> factual test temperatures (xarray dataset) lat: 32, lon: 32, time: 14307
    # x_te_reduced_eth_fact -> land grid cells factual temperature data
    # x_te_reduced_eth_cf   -> land grid cells counterfactual temperature data
    #print("x_te_reduced_eth_fact:", x_te_reduced_eth_fact.shape)

    # datasets
    ds_test_1300_eth_fact = ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1])) # HERE TP
    #print("ds_test_1300_eth_fact:", ds_test_1300_eth_fact)
    ds_test_1300_eth_cf = ds_test_eth_cf.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1])) # HERE TP

    # get indices of time slices
    time_index = ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).get_index("time")
    #print("Time index:", time_index)
    indices = time_index.get_indexer(ds_test_1300_eth_fact.time.values)
    start_idx, end_idx = indices[0], indices[-1]+1 # add 1 to include last index
    #if time_period[-1] == "2100":
    #    end_idx = indices[-1]
    
    #############################

    print("Start index:", start_idx)
    print("End index:", end_idx)

    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[start_idx])
    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[end_idx])

    
    ######################
    ### Load Test Data ###
    ######################
    
    # PYTORCH arrays
    # Factual Test/True temperatures
    eth_fact_1300_test_reduced = x_te_reduced_eth_fact[:4769,:][start_idx:end_idx,:] # HERE
    eth_fact_1400_test_reduced = x_te_reduced_eth_fact[4769:2*4769,:][start_idx:end_idx,:]
    eth_fact_1500_test_reduced = x_te_reduced_eth_fact[-4769:14307,:][start_idx:end_idx,:]
    print("eth_fact_1300_test_reduced shape:", eth_fact_1300_test_reduced.shape)
    mask_x_te = mask_x_te_eth_fact

    # Counterfactual
    # Factual Test/True temperatures
    eth_cf_1300_test_reduced = x_te_reduced_eth_cf[:4769,:][start_idx:end_idx,:] # HERE
    eth_cf_1400_test_reduced = x_te_reduced_eth_cf[4769:2*4769,:][start_idx:end_idx,:]
    eth_cf_1500_test_reduced = x_te_reduced_eth_cf[-4769:14307,:][start_idx:end_idx,:]
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
    
    dpa_ensemble_fact_raw = xr.open_dataset(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
    dpa_1300_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1400_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(4769,2*4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1500_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(-4769,14307)).sel(time=slice(time_period[0], time_period[1]))
    #print("dpa_1300_fact_raw:", dpa_1300_fact_raw)
    

    # shape: ensemble_member: 100, time: 14307, lat: 32, lon: 32
    dpa_ensemble_fact_restored = xr.open_dataset(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    dpa_1300_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1400_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(4769,2*4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1500_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(-4769,14307)).sel(time=slice(time_period[0], time_period[1]))
    

    # COUNTERFACTUAL
    dpa_ensemble_raw_cf = xr.open_dataset(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
    dpa_1300_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1400_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(4769,2*4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1500_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(-4769,14307)).sel(time=slice(time_period[0], time_period[1]))

    dpa_ensemble_restored_cf = xr.open_dataset(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    dpa_1300_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(0, 4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1400_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(4769,2*4769)).sel(time=slice(time_period[0], time_period[1]))
    dpa_1500_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(-4769,14307)).sel(time=slice(time_period[0], time_period[1]))
    
    #############
    ### Tests ###
    #############

    #################
    ### Quantiles ### add for random locations?? add counterfactual??
    #################

    ### Factual ###
    ###############

    quantiles = torch.linspace(0,1,21)
    log_print(log_file, f"quantiles: {quantiles}")
    
    ### TRUTH ###########################################################################################################
    # compute quantiles along 0th (time) dimension
    quantiles_fact_true = torch.quantile(eth_fact_1300_test_reduced, q=quantiles, dim=0)
    #print("quantiles shape:", quantiles_fact_true.shape)

    # compute spatial mean 
    quantiles_fact_true_spat_mean = quantiles_fact_true.mean(dim=1)

    # plot a quantile
    quantiles_fact_restored = ut.restore_nan_columns(quantiles_fact_true[2,:], mask_x_te)
    quantiles_fact_restored_xr = ut.torch_to_dataarray(quantiles_fact_restored, ds_test_eth_fact)
    fig, ax = ut.plot_map(quantiles_fact_restored_xr, np.linspace(2,6,21), cmap="YlOrRd", cmap_label = "Autocorrelation")
    #fig.savefig(f"{save_path_eth}/factual_075_quantile_truth.png")
    ######################################################################################################################

    
    ### DPA ENSEMBLE #####################################################################################################
    
    # DISTINCT members ###
    ######################
    dpa_1300_fact_raw_pt = torch.from_numpy(dpa_1300_fact_raw.values) # turn into pytorch array
    # quantiles
    quantiles_fact_dpa = torch.quantile(dpa_1300_fact_raw_pt, q=quantiles, dim=1)
    #print("DPA quantiles shape:", quantiles_fact_dpa.shape)

    # DISTINCT MEMBERS + MEAN over ens members and locations
    quantiles_fact_dpa_mean = quantiles_fact_dpa.mean(dim=(1))
    #print(quantiles_fact_dpa_mean.shape)
    quantiles_fact_dpa_mean_spat_mean = quantiles_fact_dpa_mean.mean(dim=(1))

    # DIFFERENCE DPA quantiles and true quantiles
    quantile_diffs_ens = torch.abs(quantiles_fact_dpa - quantiles_fact_true.unsqueeze(1))

    # difference: mean over ensemble members and locations (grid-cells)
    # the absolute errors of the sampled vs. true RCM’s quantiles are averaged across all locations
    quantile_diffs_ens_mean = quantile_diffs_ens.mean(dim=(1,2)) # absolute errors averaged across ensemble members and locations
    #print("quantile diffs shape", quantile_diffs_ens.shape)
    log_print(log_file, f"Factual mean quantile differences (across all quantiles): {quantile_diffs_ens_mean.mean()}") # absolute errors averaged over quantiles
    log_print(log_file, f"Factual 0.05-quantile differences: {quantile_diffs_ens_mean[1]}") # absolute errors of 0.05 quantile


    # Sort both datasets
    q1 = quantiles_fact_dpa_mean_spat_mean.detach().numpy()
    q2 = quantiles_fact_true_spat_mean.detach().numpy()
    
    
    # Plot Mean of Quantiles 
    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(q1, q2, alpha=0.7, linewidths=0.2)
    ax.plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
    ax.set_ylabel("Truth quantiles")
    ax.set_xlabel("Model quantiles")
    ax.set_title("Factual Q–Q Plot: Truth vs mean of DPA ensemble quantiles spatial mean")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_path_eth}/quantiles/factual_Q-Q_plot_mean_of_quantiles_spatial_mean.png")

    
    
    
    # Quantiles of DPA ENSEMBLE MEAN ####
    #####################################
    # compute ensemble mean
    dpa_1300_fact_raw_ens_mean_pt = dpa_1300_fact_raw_pt.mean(dim=0)
    #print(dpa_1300_fact_raw_ens_mean_pt.shape)
    
    # compute quantiles 
    dpa_ens_mean_quantiles_pre = (torch.quantile(dpa_1300_fact_raw_ens_mean_pt, q=quantiles, dim=0))
    dpa_ens_mean_quantiles_spat_mean = dpa_ens_mean_quantiles_pre.mean(dim=1)
    #print(dpa_ens_mean_quantiles_spat_mean.shape)
    
    # Plot quantiles of DPA ensemble mean for spatial domain average 
    q1 = dpa_ens_mean_quantiles_spat_mean.detach().numpy()
    q2 = quantiles_fact_true_spat_mean.detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(q1, q2, alpha=0.7, linewidths=0.2)
    ax.plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
    ax.set_ylabel("Truth quantiles")
    ax.set_xlabel("Model quantiles")
    ax.set_title("Factual Q–Q Plot: Truth vs DPA ensemble mean spatial mean")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_path_eth}/quantiles/factual_Q-Q_plot_of_dpa_ens_mean_spatial_mean.png")

    
    # ENSEMBLE MEAN + SINGLE GRID CELLS ###
    # plot true against quantiles of ensemble mean 
    
    
    # Plot: Q-Q of different grid cells 
    fig, ax = plt.subplots(4,4,figsize=(16,16))
    np.random.seed(42)
    gcs = np.random.randint(0, 648, size=16)
    k = 0
    ens_memb = 21
    for i in range(4):
        for j in range(4):
            q1 = dpa_ens_mean_quantiles_pre[:,gcs[k]].detach().numpy()
            q2 = quantiles_fact_true[:,gcs[k]].detach().numpy()
            q11 = quantiles_fact_dpa[:,ens_memb,gcs[k]].detach().numpy()
            q12 = quantiles_fact_dpa_mean[:,gcs[k]].detach().numpy()

            
            ax[i, j].scatter(q1, q2, alpha=1.0, linewidths=0.2, label = "DPA Ens mean", marker = '.')
            ax[i, j].scatter(q11, q2, alpha=1.0, linewidths=0.5, label = "DPA Ens member 21", marker = 'x')
            ax[i, j].scatter(q12, q2, alpha=1.0, linewidths=0.5, label = "DPA Quantiles Ens mean", marker = 'x')

            ax[i, j].plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
            ax[i, j].set_ylabel("Truth quantiles")
            ax[i, j].set_xlabel("Model quantiles")
            ax[i, j].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Annotate upper-left corner with grid-cell ID
            ax[i, j].text(
                        0.02, 0.95,                         # position (in axes fraction)
                        f"Grid cell {gcs[k]}",              # text
                        transform=ax[i, j].transAxes,       # coordinates relative to axes
                        fontsize=10,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
                        )
            
            
            k += 1
    fig.suptitle("Factual Q–Q Plots of DPA Ensemble Mean vs Truth (16 Random Grid Cells)", fontsize=16) #y=0.92)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{save_path_eth}/quantiles/factual_Q-Q_plot_of_dpa_ens_mean_single_grid_cells.png")
    
    ######################################################################################################################

    ### Counterfactual ###
    ######################

    ### TRUTH ###########################################################################################################
    # compute quantiles along 0th (time) dimension
    quantiles_cf_true = torch.quantile(eth_cf_1300_test_reduced, q=quantiles, dim=0)
    #print("quantiles shape:", quantiles_cf_true.shape)

    # compute spatial mean 
    quantiles_cf_true_spat_mean = quantiles_cf_true.mean(dim=1)

    # plot a quantile
    quantiles_cf_restored = ut.restore_nan_columns(quantiles_cf_true[2,:], mask_x_te)
    quantiles_cf_restored_xr = ut.torch_to_dataarray(quantiles_cf_restored, ds_test_eth_fact)
    fig, ax = ut.plot_map(quantiles_cf_restored_xr, np.linspace(2,6,21), cmap="YlOrRd", cmap_label = "Autocorrelation")
    #fig.savefig(f"{save_path_eth}/quantiles/counterfactual_075_quantile_truth.png")
    ######################################################################################################################

    # DISTINCT members ###
    ######################
    dpa_1300_cf_raw_pt = torch.from_numpy(dpa_1300_cf_raw.values) # turn into pytorch array
    # quantiles
    quantiles_cf_dpa = torch.quantile(dpa_1300_cf_raw_pt, q=quantiles, dim=1)
    #print("DPA quantiles shape:", quantiles_fact_dpa.shape)

    # DISTINCT MEMBERS + MEAN over ens members and locations
    quantiles_cf_dpa_mean = quantiles_cf_dpa.mean(dim=(1))
    #print(quantiles_cf_dpa_mean.shape)
    quantiles_cf_dpa_mean_spat_mean = quantiles_cf_dpa_mean.mean(dim=(1))

    # DIFFERENCE DPA quantiles and true quantiles
    quantile_diffs_ens_cf = torch.abs(quantiles_cf_dpa - quantiles_cf_true.unsqueeze(1))

    # difference: mean over ensemble members and locations (grid-cells)
    # the absolute errors of the sampled vs. true RCM’s quantiles are averaged across all locations
    quantile_diffs_ens_mean_cf = quantile_diffs_ens_cf.mean(dim=(1,2)) # absolute errors averaged across ensemble members and locations
    #print("quantile diffs shape", quantile_diffs_ens_cf.shape)
    log_print(log_file, f"Counterfactual mean quantile differences (across all quantiles): {quantile_diffs_ens_mean_cf.mean()}") # absolute errors averaged over quantiles
    log_print(log_file, f"Counterfactual 0.05-quantile differences: {quantile_diffs_ens_mean_cf[1]}") # absolute errors of 0.05 quantile


    # Sort both datasets
    q1 = quantiles_cf_dpa_mean_spat_mean.detach().numpy()
    q2 = quantiles_cf_true_spat_mean.detach().numpy()
    
    
    # Plot Mean of Quantiles 
    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(q1, q2, alpha=0.7, linewidths=0.2)
    ax.plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
    ax.set_ylabel("Truth quantiles")
    ax.set_xlabel("Model quantiles")
    ax.set_title("Counterfactual Q–Q Plot: Truth vs Mean of Ensemble quantiles spatial mean")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_path_eth}/quantiles/counterfactual_Q-Q_plot_mean_of_quantiles_spatial_mean.png")


    # Quantiles of DPA ENSEMBLE MEAN ####
    #####################################
    # compute ensemble mean
    dpa_1300_cf_raw_ens_mean_pt = dpa_1300_cf_raw_pt.mean(dim=0)
    #print(dpa_1300_cf_raw_ens_mean_pt.shape)
    
    # compute quantiles 
    dpa_ens_mean_quantiles_pre_cf = (torch.quantile(dpa_1300_cf_raw_ens_mean_pt, q=quantiles, dim=0))
    dpa_ens_mean_quantiles_spat_mean_cf = dpa_ens_mean_quantiles_pre_cf.mean(dim=1)
    #print(dpa_ens_mean_quantiles_spat_mean_cf.shape)
    
    # Plot quantiles of DPA ensemble mean for spatial domain average 
    q1 = dpa_ens_mean_quantiles_spat_mean_cf.detach().numpy()
    q2 = quantiles_cf_true_spat_mean.detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(q1, q2, alpha=0.7, linewidths=0.2)
    ax.plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
    ax.set_ylabel("Truth quantiles")
    ax.set_xlabel("Model quantiles")
    ax.set_title("Counterfactual Q–Q Plot: Truth vs DPA Ensemble mean quantiles spatial mean")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_path_eth}/quantiles/counterfactual_Q-Q_plot_of_dpa_ens_mean_spatial_mean.png")

    
    # ENSEMBLE MEAN + SINGLE GRID CELLS ###
    # plot true against quantiles of ensemble mean 
    
    
    # Plot: Q-Q of different grid cells 
    fig, ax = plt.subplots(4,4,figsize=(16,16))
    np.random.seed(42)
    gcs = np.random.randint(0, 648, size=16)
    k = 0
    ens_memb = 21
    for i in range(4):
        for j in range(4):
            q1 = dpa_ens_mean_quantiles_pre_cf[:,gcs[k]].detach().numpy()
            q2 = quantiles_cf_true[:,gcs[k]].detach().numpy()
            q11 = quantiles_cf_dpa[:,ens_memb,gcs[k]].detach().numpy()
            q12 = quantiles_cf_dpa_mean[:,gcs[k]].detach().numpy()

            
            ax[i, j].scatter(q1, q2, alpha=1.0, linewidths=0.2, label = "DPA Ens mean", marker = '.')
            ax[i, j].scatter(q11, q2, alpha=1.0, linewidths=0.5, label = "DPA Ens member 21", marker = 'x')
            ax[i, j].scatter(q12, q2, alpha=1.0, linewidths=0.5, label = "DPA Quantiles Ens mean", marker = 'x')

            ax[i, j].plot([q1.min(), q1.max()], [q1.min(), q1.max()], 'r--', label='1:1 line')
            ax[i, j].set_ylabel("Truth quantiles")
            ax[i, j].set_xlabel("Model quantiles")
            ax[i, j].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Annotate upper-left corner with grid-cell ID
            ax[i, j].text(
                        0.02, 0.95,                         # position (in axes fraction)
                        f"Grid cell {gcs[k]}",              # text
                        transform=ax[i, j].transAxes,       # coordinates relative to axes
                        fontsize=10,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
                        )
            
            
            k += 1
    fig.suptitle("Counterfactual Q–Q Plots of DPA Ensemble Mean vs Truth (16 Random Grid Cells)", fontsize=16) #y=0.92)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{save_path_eth}/quantiles/counterfactual_Q-Q_plot_of_dpa_ens_mean_single_grid_cells.png")
    
    
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
    
    ### Factual ###
    ###############

    # dpa ensemble for one grid cell list[(time steps,1), (time steps,1), ...]
    # this is the reconstructed array containing 1024 grid cells (many with NaNs)
    _, dpa_list, _, _, _ = ut.load_dpa_arrays(path=f"{ensemble_path}/",
                                  mask = mask_x_te,
                                  ds_coords = ds_test_eth_fact,
                                  ens_members=ens_members)
    #print("DPA tensor list:", len(dpa_list))
    #print("DPA tensor list element shape:", dpa_list[0].shape) # list elements contain all timesteps 14307 (3 x 4769)

    
    print("Now calculating spatial energy loss ...")
    e_loss_array = torch.zeros(3,648)
    for i in range(648):
        # keep only subset of tensors in list
        dpa_list_subset = [t[start_idx:end_idx, i] for t in dpa_list] #[t[:4769, i] for t in dpa_list]
    
        ### calculate energy score ###
        e_loss = energy_loss(eth_fact_1300_test_reduced[:,i], dpa_list_subset)
        e_loss_array[0,i] = e_loss[0].detach() 
        e_loss_array[1,i] = e_loss[1].detach()
        e_loss_array[2,i] = e_loss[2].detach()
        #print(i)
        #print("Energy Loss:", e_loss)

    print("E Loss shape:", e_loss_array.shape)
    
    #print("type mask:", type(mask_x_te))
    #print("mask shape:", mask_x_te.shape)
    #print("type e loss array:", type(e_loss_array))

    
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array[0,:].unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(e_loss_array[1,:].unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(e_loss_array[2,:].unsqueeze(0), mask_x_te)

    #print("e_loss_restored shape:", e_loss_restored.shape)

    

    # write to logfile
    log_print(log_file, f"Factual Energy-Score spatial mean: {e_loss_array[0,:].mean()}")
    log_print(log_file, f"Factual S1-Score spatial mean: {e_loss_array[1,:].mean()}")
    log_print(log_file, f"Factual S2-Score spatial mean: {e_loss_array[2,:].mean()}")
    
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

    
    ### Counterfactual ###
    ######################
    
    # climate_list: ["cf_gen", "gen"]
    _, dpa_list_cf_pre, _, _, _ = ut.load_both_dpa_arrays(path=f"{ensemble_path}/",
                                  mask = mask_x_te,
                                  ds_coords = ds_test_eth_fact,
                                  ens_members=ens_members,
                                  climate_list = ["cf_gen"]
                                )

    #print("DPA tensor list:", len(dpa_list_cf_pre))
    #print("DPA tensor list element shape:", dpa_list_cf_pre[0][0].shape) # list elements contain all timesteps 14307 (3 x 4769)

    dpa_list_cf = dpa_list_cf_pre[0]
    
    e_loss_array = torch.zeros(3,648)
    for i in range(648):
        # keep only subset of tensors in list
        dpa_list_subset = [t[start_idx:end_idx, i] for t in dpa_list_cf]
    
        ### calculate energy score ###
        e_loss = energy_loss(eth_cf_1300_test_reduced[:,i], dpa_list_subset)
        #print(e_loss)
        e_loss_array[0,i] = e_loss[0].detach() 
        e_loss_array[1,i] = e_loss[1].detach()
        e_loss_array[2,i] = e_loss[2].detach()
        #print(i)
        #print("Energy Loss:", e_loss)
    
    print("Spatial energy loss calculated ...")
    print("E Loss shape:", e_loss_array.shape)
    
    #print("type mask:", type(mask_x_te))
    #print("mask shape:", mask_x_te.shape)
    #print("type e loss array:", type(e_loss_array))

    # Check if there is any NaN at all
    has_nan = torch.isnan(e_loss_array).any()
    #print("E_Loss has any nans:", has_nan)
        
    # restore NaN columns
    e_loss_restored = ut.restore_nan_columns(e_loss_array[0,:].unsqueeze(0), mask_x_te)
    s1_loss_restored = ut.restore_nan_columns(e_loss_array[1,:].unsqueeze(0), mask_x_te)
    s2_loss_restored = ut.restore_nan_columns(e_loss_array[2,:].unsqueeze(0), mask_x_te)

    # write to logfile
    log_print(log_file, f"Counterfactual Energy-Score spatial mean: {e_loss_array[0,:].mean()}")
    log_print(log_file, f"Counterfactual S1-Score spatial mean: {e_loss_array[1,:].mean()}")
    log_print(log_file, f"Counterfactual S2-Score spatial mean: {e_loss_array[2,:].mean()}")

    #print("e_loss_restored shape:", e_loss_restored.shape)

    
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

    # now transpose (switch i and :) everything to calculate scores over time i.e. for each map
    print("Now calculating time resolved energy loss ...")

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
            #print(i)
            #print("Energy Loss:", e_loss)

        # Save
        torch.save(e_loss_array, f"{ensemble_path}/e_loss_over_time.pt")

    else:
        e_loss_array = torch.load(f"{ensemble_path}/e_loss_over_time.pt")
    #print("e loss shape:", e_loss_pre.shape)
    print("Time resolved energy loss calculated ...")
    # create xarray from e_loss
    loss_types = ["energy_loss", "S1", "S2"]

    # Create time coordinate (example: daily timestamps)
    time_coord = ds_test_eth_fact.time
    
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

    # standardize loss values
    # Standardize along the 'time' dimension
    eloss_standardized = (eloss_xr_time_series - eloss_xr_time_series.mean(dim="time")) / eloss_xr_time_series.std(dim="time")
    #print(eloss_standardized)

    # compute yearly mean values for energy loss
    yearly_eloss_standardized = (eloss_standardized.groupby("time.year").mean(dim="time"))

    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize_ts)
    
    # Plot standardized... 
    # energy loss (yearly)
    yearly_eloss_standardized.sel(loss="energy_loss").plot(
        ax=ax,
        label="Energy Loss (Standardized)",
        color="tab:blue",
        linewidth=1
    )

    # S1
    yearly_eloss_standardized.sel(loss="S1").plot(
        ax=ax,
        label="S1",
        color="tab:orange",
        linewidth=1
    )

    # S2
    yearly_eloss_standardized.sel(loss="S2").plot(
        ax=ax,
        label="S2",
        color="tab:green",
        linewidth=1
    )
    
    # Add legend and labels
    ax.legend(fontsize=10)
    ax.set_title("Factual Test Losses", fontsize=title_fontsize)
    ax.set_xlabel("Year", fontsize=title_fontsize)
    ax.set_ylabel("Loss (Standardized)", fontsize=title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    fig.savefig(f"{save_path_eth}/energy_loss_time_series_eth_test_set.png", dpi=300)
    plt.show()

    ### Counterfactual ###
    ######################

    # now transpose (switch i and :) everything to calculate scores over time i.e. for each map
    
    if True:
        
        e_loss_array = torch.zeros(dpa_list_cf[0].shape[0], 3)
        for i in range(dpa_list_cf[0].shape[0]):
            # keep only subset of tensors in list
            dpa_list_subset = [t[i, :] for t in dpa_list_cf] 
        
            ### calculate energy score ###
            e_loss = energy_loss(x_te_reduced_eth_cf[i,:], dpa_list_subset)
            e_loss_array[i,0] = e_loss[0].detach() 
            e_loss_array[i,1] = e_loss[1].detach()
            e_loss_array[i,2] = e_loss[2].detach()
            print(i)
            print("Energy Loss:", e_loss)

        # Save
        torch.save(e_loss_array, f"{ensemble_path}/e_loss_over_time_cf.pt")
    else:
        e_loss_array = torch.load(f"{ensemble_path}/e_loss_over_time_cf.pt")
    #print("e loss shape:", e_loss_pre.shape)

    # create xarray from e_loss
    loss_types = ["energy_loss", "S1", "S2"]

    # Create time coordinate (example: daily timestamps)
    time_coord = ds_test_eth_fact.time
    
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

    # standardize loss values
    # Standardize along the 'time' dimension
    eloss_standardized = (eloss_xr_time_series - eloss_xr_time_series.mean(dim="time")) / eloss_xr_time_series.std(dim="time")
    #print(eloss_standardized)

    # compute yearly mean values for energy loss
    yearly_eloss_standardized = (eloss_standardized.groupby("time.year").mean(dim="time"))

    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize_ts)
    
    # Plot standardized... 
    # energy loss (yearly)
    yearly_eloss_standardized.sel(loss="energy_loss").plot(
        ax=ax,
        label="Energy Loss (Standardized)",
        color="tab:blue",
        linewidth=1
    )

    # S1
    yearly_eloss_standardized.sel(loss="S1").plot(
        ax=ax,
        label="S1",
        color="tab:orange",
        linewidth=1
    )

    # S2
    yearly_eloss_standardized.sel(loss="S2").plot(
        ax=ax,
        label="S2",
        color="tab:green",
        linewidth=1
    )
    
    # Add legend and labels
    ax.legend(fontsize=10)
    ax.set_title("Counterfactual Test Losses", fontsize=title_fontsize)
    ax.set_xlabel("Year", fontsize=title_fontsize)
    ax.set_ylabel("Loss (Standardized)", fontsize=title_fontsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    fig.savefig(f"{save_path_eth}/energy_loss_time_series_eth_test_set_cf.png", dpi=300)
    plt.show()
    
    
    
    
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

    ### Factual ###
    ###############

    # mean of RAW factual ensemble
    #print("Calculating ensemble mean ...")
    #dpa_fact_1300_raw_mean = dpa_ens_mean_fact_1300_raw.mean(dim="ensemble_member")

    ## turn dpa ens (mean) into torch array
    dpa_ens_mean_fact_1300_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1300_raw.values) #dpa_ens_mean_pt
    
    # check shapes and types
    #print("dpa_ens_mean_pt shape", dpa_ens_mean_fact_1300_raw_pt.shape)

    # calculate correlation
    #print("Now calculating pearson correlation")
    # test temperature vs. eg dpa_ens_mean_1300
    r_cols = evaluation.pearsonr_cols(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)  # shape: (648,)

    #print("Correlation calculated ...")
    #print(type(r_cols))
    
    # prepare for plotting
    
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    corr_spatial_pre = ut.restore_nan_columns(r_cols[None, :], mask_x_te)
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
    dpa_ens_mean_cf_1300_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1300_raw.values) #dpa_ens_mean_pt

    # calculate correlation
    r_cols_cf = evaluation.pearsonr_cols(eth_cf_1300_test_reduced, dpa_ens_mean_cf_1300_raw_pt, dim=0)  # shape: (648,)

    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    corr_spatial_pre_cf = ut.restore_nan_columns(r_cols_cf[None, :], mask_x_te)

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

    r2 = evaluation.r2_score(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)

    # plot
    r2_spatial_pre = ut.restore_nan_columns(r2[None, :], mask_x_te)
    #print("R2 spatial", r2_spatial_pre.shape)
    
    # turn statistics array into xarray
    r2_spatial = ut.torch_to_dataarray(r2_spatial_pre, ds_test, name="R^2")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_r2 = ut.plot_temperature_panel(ax, r2_spatial, r2_spatial.max().item(), levels = np.linspace(0.7, 1, 7), cbar=True, cmap="YlOrRd")
    ax.set_title("R2", fontsize=title_fontsize)

    plt.savefig(f"{save_path_eth}/R2_eth_test_figure.png")

    

    #############################
    ### Reliability Index Map ###
    #############################

    ### Factual ###
    ###############
    
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
    #print("RI spatial shape", ri_spatial_pre.shape)
    log_print(log_file, f"Factual RI spatial mean: {ri_vals.mean()}")
    
    # turn statistics array into xarray
    ri_spatial = ut.torch_to_dataarray(ri_spatial_pre, ds_test, name="Reliability Index")
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_map, subplot_kw={'projection': ccrs.PlateCarree()})
    plot_ri = ut.plot_temperature_panel(ax, ri_spatial, ri_spatial.max().item(), levels = np.linspace(0, 0.3, 13), cbar=True)
    ax.set_title("Factual Reliability Index", fontsize = title_fontsize)
    plt.savefig(f"{save_path_eth}/factual_ri_spatial.png")

    ### Counterfactual ###
    ######################

    # turn array into .pt array
    dpa_1300_cf_raw_pt = torch.from_numpy(dpa_1300_cf_raw.values)

    # array to store reliability index
    ri_vals_cf = torch.zeros((648))

    for n in range(648):
        # INSERT FULL DPA ENSEMBLE IN FOLLOWING LINE
        ranks = ut.rank_histogram(eth_cf_1300_test_reduced[:,n].detach().numpy(), dpa_1300_cf_raw_pt[:-1,:,n].T.detach().numpy()) # dpa_ens_pt
        counts, bin_edges = np.histogram(
            ranks, 
            bins=np.arange(ens_members + 2) - 0.5
            )
    
        # reliability index
        ri_cf = ut.reliability_index(counts)
        ri_vals_cf[n] = ri_cf

    # PLOT
    # restore nans
    # add dimension for function ut.restore_nan_columns to work correctly
    ri_spatial_cf_pre = ut.restore_nan_columns(ri_vals_cf[None, :], mask_x_te)
    
    print("RI spatial shape", ri_spatial_cf_pre.shape)
    log_print(log_file, f"Counterfactual RI spatial mean: {ri_vals_cf.mean()}")
    
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

    for year in years:
        ### Factual ###
        fig, ax = evaluation.plot_dpa_time_series(true_t = ds_test_1300_eth_fact, 
                                                  dpa_ens = dpa_1300_fact_restored, 
                                                  dpa_ens_mean = dpa_ens_mean_fact_1300_restored,  
                                                  lat_min = ger_lat_min, 
                                                  lat_max = ger_lat_max, 
                                                  lon_min = ger_lon_min, 
                                                  lon_max = ger_lon_max, 
                                                  plot_year = year,
                                                  figsize_ts = figsize_ts,
                                                  title_fontsize = title_fontsize,
                                                  title = f"Factual Temperatures Germany {year}", 
                                                  climate = "Factual"
                                                  )        
        fig.savefig(f"{save_path_eth}/Germany/Germany_mean_fact_T_ts_{year}.png")
        plt.show()
    
        ### Counterfactual ###
        fig, ax = evaluation.plot_dpa_time_series(true_t = ds_test_1300_eth_cf, 
                                                  dpa_ens = dpa_1300_cf_restored, 
                                                  dpa_ens_mean = dpa_ens_mean_cf_1300_restored, 
                                                  lat_min = ger_lat_min, 
                                                  lat_max = ger_lat_max, 
                                                  lon_min = ger_lon_min, 
                                                  lon_max = ger_lon_max, 
                                                  plot_year = year,
                                                  figsize_ts = figsize_ts,
                                                  title_fontsize = title_fontsize,
                                                  title = f"Counterfactual Temperatures Germany {year}",
                                                  climate = "Counterfactual")        
        fig.savefig(f"{save_path_eth}/Germany/Germany_mean_cf_T_ts_{year}.png")
        plt.show()
    
    
    ### Spain ###

    # coordinates
    sp_lat_min = 38
    sp_lat_max = 42
    sp_lon_min = -8
    sp_lon_max = 0

    for year in years:
        ### Factual ###
        fig, ax = evaluation.plot_dpa_time_series(true_t = ds_test_1300_eth_fact, 
                                                  dpa_ens = dpa_1300_fact_restored, 
                                                  dpa_ens_mean = dpa_ens_mean_fact_1300_restored,  
                                                  lat_min = sp_lat_min, 
                                                  lat_max = sp_lat_max, 
                                                  lon_min = sp_lon_min, 
                                                  lon_max = sp_lon_max, 
                                                  plot_year = year,
                                                  figsize_ts = figsize_ts,
                                                  title_fontsize = title_fontsize,
                                                  title = f"Factual Temperatures Spain {year}", 
                                                  climate = "Factual"
                                                  )        
        fig.savefig(f"{save_path_eth}/Spain/Spain_mean_fact_T_ts_{year}.png")
        plt.show()

        ### Counterfactual ###
        fig, ax = evaluation.plot_dpa_time_series(true_t = ds_test_1300_eth_cf, 
                                                  dpa_ens = dpa_1300_cf_restored, 
                                                  dpa_ens_mean = dpa_ens_mean_cf_1300_restored, 
                                                  lat_min = sp_lat_min, 
                                                  lat_max = sp_lat_max, 
                                                  lon_min = sp_lon_min, 
                                                  lon_max = sp_lon_max, 
                                                  plot_year = year,
                                                  figsize_ts = figsize_ts,
                                                  title_fontsize = title_fontsize,
                                                  title = f"Counterfactual Temperatures Spain {year}",
                                                  climate = "Counterfactual")        
        fig.savefig(f"{save_path_eth}/Spain/Spain_mean_cf_T_ts_{year}.png")
        plt.show()
    
    
    
                          

    ### True (Test) temperature ###
    # true temperature - germany spatial average
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
    
    
    # plot
    #n_stds = 2
    #lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    #upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger

    #print("dpa_ens_mean_ger:", dpa_ens_mean_ger)
    #print("lower env:", lower_env)
    #print("upper env:", upper_env)
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    #year = "2040"
    #temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="Factual Truth")
    #dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA factual mean")
    
    # ADD COUNTERFACTUAL TEMPERATURE HERE
    #cf_temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="CF Truth")
    #dpa_ens_mean_ger_cf.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA CF mean")
    # ################################## #
    
    
    # fill_between needs numpy arrays + axis
    #ax.fill_between(
    #    dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values,   # x-axis values (datetime64)
    #    lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
    #    upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
    #    color="tab:orange", alpha=0.2, label=f"+/- {n_stds} std"
    #)

    
    
    #ax.legend()
    #plt.tight_layout()
    #ax.set_title("DPA mean-Temperature in Germany Domain", fontsize=title_fontsize)
    #fig.savefig("ETH_analysis_results/Germany_mean_T_ts.png")
    #plt.show()

    ####################
    ### Violin plots ###
    ####################

    ### max temperatures ###
    fig = evaluation.plot_violins(truth = ds_test_1300_eth_fact, 
                            dpa_ensemble = dpa_1300_fact_restored,
                            save_path = f"{save_path_eth}",
                            lat_min = ger_lat_min,
                            lat_max = ger_lat_max,
                            lon_min = ger_lon_min,
                            lon_max = ger_lon_max
                            )



    fig = evaluation.plot_violins(truth = ds_test_1300_eth_cf, 
                            dpa_ensemble = dpa_1300_cf_restored,
                            save_path = f"{save_path_eth}",
                            lat_min = ger_lat_min,
                            lat_max = ger_lat_max,
                            lon_min = ger_lon_min,
                            lon_max = ger_lon_max,
                            in_fact_for_cf = ds_test_1300_eth_fact
                            )


    ### random temperatures ###
    fig = evaluation.plot_violins(truth = ds_test_1300_eth_fact, 
                            dpa_ensemble = dpa_1300_fact_restored,
                            save_path = f"{save_path_eth}",
                            lat_min = ger_lat_min,
                            lat_max = ger_lat_max,
                            lon_min = ger_lon_min,
                            lon_max = ger_lon_max,
                            mode="random"
                            )



    fig = evaluation.plot_violins(truth = ds_test_1300_eth_cf, 
                            dpa_ensemble = dpa_1300_cf_restored,
                            save_path = f"{save_path_eth}",
                            lat_min = ger_lat_min,
                            lat_max = ger_lat_max,
                            lon_min = ger_lon_min,
                            lon_max = ger_lon_max,
                            in_fact_for_cf = ds_test_1300_eth_fact,
                            mode="random"
                            )

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
    


    ############
    ### BIAS ###
    ############

    ### Europe domain spatial average ###

    
    # reshape mask
    mask_xr = ut.torch_to_dataarray(mask_x_te.unsqueeze(0), ds, lat_dim=32, lon_dim=32, name="Mask") # mask_x_te.unsqueeze(0) adds dimension for torch_to_dataarray function to work properly


    ### DPA Ensemble ###
    dpa_ens_mean_entire_domain = dpa_1300_fact_raw.mean(dim = ("ensemble_member", "lat_x_lon"), skipna=True) #dpa_ensemble_raw
    #print(dpa_ens_mean_entire_domain)

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
    #print(reshaped.shape) # shape = (32, 32, 100, 4769)
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
    #print(arr_unstacked)
    le_T_mean_spat_mean = arr_unstacked.mean(dim=("ensemble_member", "lat", "lon"), skipna=True)

    ### compute yearly means ###
    yearly_mean_dpa = dpa_ens_mean_entire_domain.groupby("time.year").mean()
    yearly_mean_le = le_T_mean_spat_mean.groupby("time.year").mean()


    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)
    #dpa_ens_mean_entire_domain.plot(ax=ax, label="Europe DPA Ensemble Mean")
    #le_T_mean_spat_mean.plot(ax=ax, label="Europe LE Forced Response")

    yearly_mean_dpa.plot(ax=ax, label="Europe DPA Ensemble Mean")
    yearly_mean_le.plot(ax=ax, label="Europe LE Forced Response")

    #print("dpa_ens_mean_entire_domain:", dpa_ens_mean_entire_domain)
    #print("le_T_mean_spat_mean:", le_T_mean_spat_mean)
    
    ax.legend()
    ax.set_title("European Domain Average Yearly Average Temperature", fontsize=title_fontsize)
    ax.set_xlabel("Year", fontsize=title_fontsize)
    ax.set_ylabel("TREFHT", fontsize=title_fontsize)
    fig.savefig(f"{save_path_eth}/LE_Europe_Forced_response.png")
    plt.show()


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
    
