import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
#from torchvision.utils import make_grid
from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss_two_sample
import argparse
import json
import xarray as xr
import torch.nn as nn
#from sklearn.manifold import TSNE

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#########################
### ACTUALLY REQUIRED ###
#########################

def remove_nan_columns(tensor: torch.Tensor):
    """
    Removes columns that are entirely NaN from a 2D tensor.
    
    Returns:
        - reduced_tensor: Tensor with NaN-only columns removed
        - column_mask: Boolean mask indicating which columns were kept
    """
    if tensor.dim() != 2:
        raise ValueError("Input must be a 2D tensor")
        
    column_mask = ~torch.isnan(tensor).all(dim=0)
    reduced_tensor = tensor[:, column_mask]
    return reduced_tensor, column_mask

def data_to_torch(ds, variable):
    if isinstance(ds, xr.Dataset):
        temp_data = ds[variable]
    else:
        temp_data = ds
    data = temp_data.transpose('time', 'lat', 'lon')
    
    # Now convert to numpy
    data_np = data.values  # Shape: (time, lat, lon)
    
    # Flatten lat and lon together
    time_steps, lat_dim, lon_dim = data_np.shape
    data_np = data_np.reshape(time_steps, lat_dim * lon_dim)
    
    
    data_np = data_np  # Shape: (grid_cell, timestep)
    
    # Finally, convert to torch tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    print(data_tensor.shape)
    return data_tensor

def standardize_numpy(X, mean=None, std=None):
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be both provided or both None")

    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)

    return (X - mean) / std, mean, std

def plot_all_losses(
    loss_total_tr, loss_total_te, loss_s1_tr_total, loss_s2_tr_total,
    loss_s1_te_total, loss_s2_te_total,
    loss_dpa_tr, loss_dpa_te, loss_s1_dpa_tr, loss_s2_dpa_tr,
    loss_s1_dpa_te, loss_s2_dpa_te,
    loss_lin_lat_pred_tr, loss_lin_lat_pred_te,
    loss_s1_lin_lat_pred_tr, loss_s2_lin_lat_pred_tr,
    loss_s1_lin_lat_pred_te, loss_s2_lin_lat_pred_te,
    training_epochs):
    
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))

    # --- Subplot 1: Combined Losses ---
    axs[0].plot(loss_total_tr, label="Train", color='tab:blue')
    axs[0].plot(loss_total_te, label="Test", color='tab:orange')
    axs[0].plot(loss_s1_tr_total, label="Train: S1 Total Loss", color='tab:green')
    axs[0].plot(loss_s2_tr_total, label="Train: S2 Total Loss", color='tab:red')
    axs[0].plot(loss_s1_te_total, label="Test: S1 Total Loss", color='tab:brown')
    axs[0].plot(loss_s2_te_total, label="Test: S2 Total Loss", color='tab:purple')
    axs[0].set_title("Total Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(0, training_epochs)

    # --- Subplot 2: DPA Losses ---
    axs[1].plot(loss_dpa_tr, label="Train", color='tab:red')
    axs[1].plot(loss_dpa_te, label="Test", color='tab:green')
    axs[1].plot(loss_s1_dpa_tr, label="Train S1", color='tab:blue')
    axs[1].plot(loss_s2_dpa_tr, label="Train S2", color='tab:orange')
    axs[1].plot(loss_s1_dpa_te, label="Test S1", color='tab:brown')
    axs[1].plot(loss_s2_dpa_te, label="Test S2", color='tab:purple')
    axs[1].set_title("DPA Losses")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(0, training_epochs)

    # --- Subplot 3: Latent Map Losses ---
    axs[2].plot(loss_lin_lat_pred_tr, label="Train", color='tab:red')
    axs[2].plot(loss_lin_lat_pred_te, label="Test", color='tab:green')
    axs[2].plot(loss_s1_lin_lat_pred_tr, label="Train S1", color='tab:blue')
    axs[2].plot(loss_s2_lin_lat_pred_tr, label="Train S2", color='tab:orange')
    axs[2].plot(loss_s1_lin_lat_pred_te, label="Test S1", color='tab:brown')
    axs[2].plot(loss_s2_lin_lat_pred_te, label="Test S2", color='tab:purple')
    axs[2].set_title("Latent Map Losses")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlim(0, training_epochs)

    plt.tight_layout()
    plt.show()


def load_both_dpa_arrays(path, mask, ds_coords, ens_members, save_path=None, no_epochs="not_specified", climate_list=[]):
    '''
    climate_list: ["cf_gen", "gen"]: list of climates (counterfactual and factual) to laod and save
    
    returns:
        lists containing [counterfactual, factual] arrays
    '''
    list_tensor_list = []
    list_tensor_list_raw = []
    list_stacked = []
    list_stacked_reshaped = []
    list_ds = []

    for climate in climate_list:
        print("Now loading DPA ensemble restored including nans")
        print("Path:", f"{path}{climate}1_te.pt")
        tensor_list_raw = [torch.load(f"{path}{climate}{i}_te.pt", map_location=torch.device('cpu'), weights_only=False) for i in range(1,ens_members+1)] #this is loading raw arrays without nans
        list_tensor_list_raw.append(tensor_list_raw)
        print("Tensor list raw length:", len(tensor_list_raw))
        print("Tensor list raw elements shape:", tensor_list_raw[0].shape)
        stacked_raw = torch.stack(tensor_list_raw)  # shape: (N, T, H, W)
        stacked_reshaped_raw = stacked_raw.reshape(stacked_raw.shape[0], stacked_raw.shape[1], stacked_raw.shape[2])
        print((stacked_raw.shape[0], stacked_raw.shape[1], stacked_raw.shape[2]))
    
    
        
        # create dataset
        ds_raw = xr.Dataset({
                        "TREFHT": xr.DataArray(stacked_reshaped_raw.detach().numpy(), 
                             dims=("ensemble_member", "time", "lat_x_lon"),
                             coords={
                                    "ensemble_member": np.arange(1,stacked_reshaped_raw.shape[0]+1),
                                    "time": ds_coords.time,
                                    "lat_x_lon": np.arange(stacked_raw.shape[2])
                                    },
                                              )
        })
        print(ds_raw)
        if save_path is not None:
            ds_raw.to_netcdf(f"{save_path}/raw_ETH_{climate}_dpa_ens_{no_epochs}_dataset.nc", format="NETCDF4")
            
        ### Restore NaNs ###
        
        # Load and stack into one tensor
        print("Now loading DPA ensemble restored including nans")
        tensor_list = [restore_nan_columns(torch.load(f"{path}{climate}{i}_te.pt", map_location=torch.device('cpu'), weights_only=False), mask) for i in range(1,ens_members+1)] #this is loading raw arrays without nans
        list_tensor_list.append(tensor_list)
        print("Tensor list length:", len(tensor_list))
        print("Tensor list elements shape:", tensor_list[0].shape)
        stacked = torch.stack(tensor_list)  # shape: (N, T, H, W)
        list_stacked.append(stacked)
        stacked_reshaped = stacked.reshape(stacked.shape[0], stacked.shape[1], 32, 32)
        list_stacked_reshaped.append(stacked_reshaped)
    
    
        
        # create dataset
        ds = xr.Dataset({
                        "TREFHT": xr.DataArray(stacked_reshaped.detach().numpy(), 
                             dims=("ensemble_member", "time", "lat", "lon"),
                             coords={
                                    "ensemble_member": np.arange(1,stacked_reshaped.shape[0]+1),
                                    "time": ds_coords.time,
                                    "lat": ds_coords.lat,
                                    "lon": ds_coords.lon
                                    },
                                              )
        })
        list_ds.append(ds)
        print(ds)
        if save_path is not None:
            ds.to_netcdf(f"{save_path}/ETH_{climate}_dpa_ens_{no_epochs}_dataset_restored.nc", format="NETCDF4")

    return list_tensor_list, list_tensor_list_raw, list_stacked, list_stacked_reshaped, list_ds

    
def restore_nan_columns(reduced_tensor: torch.Tensor, column_mask: torch.Tensor):
    """
    Restores removed NaN-only columns to a tensor using the original column mask.
    
    Returns:
        - restored_tensor: Tensor with original number of columns (NaNs in removed positions)
    """
    if reduced_tensor.ndim == 1:
        n_rows = 1
    elif reduced_tensor.ndim > 1:
        n_rows = reduced_tensor.shape[0]
    #n_rows = reduced_tensor.shape[0]
    n_cols = column_mask.numel()
    
    restored_tensor = torch.full((n_rows, n_cols), float('nan'), dtype=reduced_tensor.dtype, device=reduced_tensor.device)
    restored_tensor[:, column_mask] = reduced_tensor
    return restored_tensor

