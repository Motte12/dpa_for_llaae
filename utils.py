import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torchvision.utils import make_grid
from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss_two_sample
import argparse
import json
import xarray as xr
import torch.nn as nn
from sklearn.manifold import TSNE

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    
def restore_nan_columns(reduced_tensor: torch.Tensor, column_mask: torch.Tensor):
    """
    Restores removed NaN-only columns to a tensor using the original column mask.
    
    Returns:
        - restored_tensor: Tensor with original number of columns (NaNs in removed positions)
    """
    n_rows = reduced_tensor.shape[0]
    n_cols = column_mask.numel()
    
    restored_tensor = torch.full((n_rows, n_cols), float('nan'), dtype=reduced_tensor.dtype, device=reduced_tensor.device)
    restored_tensor[:, column_mask] = reduced_tensor
    return restored_tensor

def plot_temperature_panel(ax, dataarray, vmax_shared, sample_nr=None, title=""):
    """
    Plot a single temperature panel on a given axis with Cartopy.

    Parameters:
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis on which to plot.
    dataarray : xarray.DataArray
        The temperature data to plot.
    title : str
        Title for the subplot.
    vmax_shared : float
        Symmetric max value for colormap scaling (vmin = -vmax).
    """
    levels = np.linspace(-5, 5, 11)

    p = dataarray.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        vmin=-vmax_shared,
        vmax=vmax_shared,
        levels=levels,
        add_colorbar=False
    )
    ax.set_title(title)
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    #ax.gridlines(draw_labels=False)
    if sample_nr is not None:
        ax.text(
            x=0, y=0.5,                # left edge, center vertically
            s=sample_nr,              # your annotation text
            transform=ax.transAxes,   # interpret x/y in axis coordinates (0–1)
            rotation=90,              # rotate to align with y-axis
            va='center', ha='right',  # vertical/horizontal alignment
            fontsize=12
        )
    return p

def torch_to_dataarray(x_tensor, coords_ds, lat_dim=32, lon_dim=32, name="variable"):
    """
    Convert a flattened 2D torch tensor to a 3D xarray.DataArray (lat, lon, time).

    Parameters:
    ----------
    x_tensor : torch.Tensor
        A 2D tensor of shape (time_steps, lat_dim * lon_dim), or a 1D tensor to be reshaped.
    lat_dim : int
        Number of latitude points.
    lon_dim : int
        Number of longitude points.
    coords_ds : xarray.Dataset
        Dataset containing 'lat', 'lon', and 'time' coordinates to assign.
    name : str, optional
        Name of the variable in the DataArray.

    Returns:
    -------
    xarray.DataArray
        The reshaped and labeled data as an xarray.DataArray.
    """
    # Step 1: Convert to NumPy
    data_np = x_tensor.detach().cpu().numpy()

    # Step 2: Determine time_steps and reshape
    time_steps = data_np.shape[0]
    data_np = data_np.reshape(time_steps, lat_dim, lon_dim)

    # Step 3: Transpose to (lat, lon, time)
    data_np = data_np.transpose(1, 2, 0)

    # Step 4: Create the DataArray
    da = xr.DataArray(
        data_np,
        dims=("lat", "lon", "time"),
        coords={
            "lat": coords_ds.lat,
            "lon": coords_ds.lon,
            "time": np.arange(time_steps)
        },
        name=name
    )

    return da

def standardize_numpy(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std), mean, std

def data_to_torch(ds, variable):
    temp_data = ds[variable]
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

def predictors_to_torch(ds, variable):
    temp_data = ds#[variable]
    data = temp_data.transpose('time', 'mode')
    
    # Now convert to numpy
    data_np = data.values  # Shape: (time, lat, lon)

    
    # Flatten lat and lon together
    #time_steps, lat_dim, lon_dim = data_np.shape
    #data_np = data_np.reshape(time_steps, lat_dim * lon_dim)
    
    
    #data_np = data_np  # Shape: (grid_cell, timestep)
    
    # Finally, convert to torch tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    print(data_tensor.shape)
    return data_tensor