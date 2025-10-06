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

####################################
### Spatial Evaluation functions ###
####################################
def pearsonr_cols(x, y, dim=0, eps=1e-12):
    """
    Pearson correlation per feature along `dim` (default: samples axis).
    x, y: tensors of the same shape, e.g. (n_samples, n_features)
    Returns: tensor of shape equal to the non-reduced dims (e.g. (n_features,))
    """
    # center
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean

    # numerator: covariance (without / (n-1) since it cancels in correlation)
    num = (x_c * y_c).sum(dim=dim)

    # denominator: product of std devs
    x_ss = (x_c * x_c).sum(dim=dim)
    y_ss = (y_c * y_c).sum(dim=dim)
    den = (x_ss * y_ss).sqrt().clamp_min(eps)

    return num / den

def reliability_index(counts, normalize=True):
    """
    Compute Reliability Index (RI) for a rank histogram.

    Parameters
    ----------
    counts : array-like, shape (B,)
        Bin counts of the rank histogram.
    normalize : bool, default=True
        If True, scale RI into [0, 1].

    Returns
    -------
    RI : float
        Reliability Index (0 = flat, higher = less reliable).
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    B = counts.size
    if N == 0:
        return np.nan  # undefined if no counts
    
    freqs = counts / N
    RI_raw = np.abs(freqs - 1.0/B).sum()

    if normalize:
        RI = RI_raw / (2.0 * (1 - 1.0/B))
        return RI
    else:
        return RI_raw

####################################
####################################
####################################

        
def energy_loss_per_dim_efficient(x_true, x_est, beta=1.0, verbose=True):
    """
    Per-feature energy score (one value per data_dim) with O(N M log M) time and O(N M) memory.
    Suitable for large N. Default beta=1.

    Args
    ----
    x_true : (N, D) tensor
        One true sample per data point (rows = samples, cols = features).
    x_est : list[(N, D)] or (N, M, D) tensor
        Either a list of M ensemble tensors (one (N,D) per member),
        or a stacked tensor with M in axis=1.
        Each data point i has M draws y_{i,m} from the estimated distribution.
    beta : float, default=1.0
        Energy score exponent. This routine is optimized for beta=1 (L1).
        For beta=2 we use a closed form. For other beta we fall back to a
        minibatch Monte Carlo approximation (still per-dim and memory-safe).
    verbose : bool, default=True
        If True, returns (3, D): [ES, s1, s2]; else (D,).

    Returns
    -------
    out : (D,) or (3, D) tensor
        Per-feature energy score (and optionally the two terms).
    """
    if x_true.ndim != 2:
        raise ValueError("x_true must be (N, D)")
    N, D = x_true.shape

    # Standardize x_est to (N, M, D)
    if isinstance(x_est, list):
        # stack along new axis 1 (M members)
        est = torch.stack(x_est, dim=1)  # (N, M, D)
    else:
        if x_est.ndim == 2:
            # Interpret (N*M, D) stacked vertically → split every N rows
            # (kept for backward compatibility with your original function)
            M = x_est.shape[0] // N
            est = torch.stack(list(torch.split(x_est, N, dim=0)), dim=1)  # (N, M, D)
        elif x_est.ndim == 3:
            est = x_est  # (N, M, D)
        else:
            raise ValueError("x_est must be list of (N,D), (N*M,D), or (N,M,D)")

    N2, M, D2 = est.shape
    assert N2 == N and D2 == D, "Shape mismatch between x_true and x_est"

    # Broadcast true to (N, M, D)
    true_b = x_true.unsqueeze(1).expand(N, M, D)

    if beta == 1 or abs(beta - 1.0) < 1e-12:
        # -------------------------
        # beta = 1 (L1) -> exact, efficient
        # s1 = E |X - Y|  (mean over i,m)
        s1 = (true_b - est).abs().mean(dim=(0,1))  # (D,)

        # s2 = E |Y - Y'| (mean over i of average pairwise absolute diff over m≠m')
        # For each (i,d), average absolute difference over the M ensemble members can be computed from the sorted values:
        #   AAD(i,d) = (2 / (M*(M-1))) * sum_{k=1..M} (2k - M - 1) * y_(i,k,d)
        # where y_(i,k,d) is the k-th order statistic. Then s2 = mean_i AAD(i,d).
        est_sorted, _ = torch.sort(est, dim=1)              # (N, M, D)
        # weights w_k = 2k - (M + 1), shape (M,)
        w = torch.arange(1, M+1, device=est.device, dtype=est.dtype)
        w = 2*w - (M + 1)
        # weighted sum over sorted samples → (N, D)
        aad_num = (est_sorted * w.view(1, M, 1)).sum(dim=1) # (N, D)
        aad = (2.0 / (M * (M - 1))) * aad_num               # (N, D)
        s2 = aad.mean(dim=0)                                 # (D,)

        es = s1 - 0.5 * s2

        if verbose:
            return torch.stack([es, s1, s2], dim=0)          # (3, D)
        else:
            return es                                        # (D,)

    elif beta == 2 or abs(beta - 2.0) < 1e-12:
        # -------------------------
        # beta = 2 (L2) -> closed form using second moments
        # s1 = E ||X - Y||_2^2 per-feature reduces to E[(X - Y)^2]:
        s1 = ((true_b - est) ** 2).mean(dim=(0,1))           # (D,)
        # s2 = E ||Y - Y'||_2^2 per-feature -> 2 * Var(Y):
        # average across m of (Y - mean_m Y)^2, then multiply by 2
        mu = est.mean(dim=1, keepdim=True)                   # (N,1,D)
        var = ((est - mu) ** 2).mean(dim=1)                  # (N,D)
        s2 = 2.0 * var.mean(dim=0)                           # (D,)
        es = s1 - 0.5 * s2

        if verbose:
            return torch.stack([es, s1, s2], dim=0)
        else:
            return es

    else:
        # -------------------------
        # generic beta in (0,2]: do a memory-safe Monte Carlo over pairs
        # (per-dim, without constructing (N*M x N*M)).
        # You can increase K for accuracy vs. speed; K≈min(256, M) is often enough.
        K = min(256, M)

        # s1 ≈ mean_m |X - Y|^beta
        s1 = (true_b - est).abs().pow(beta).mean(dim=(0,1))  # (D,)

        # s2 ≈ E |Y - Y'|^beta using K random pairs per (i)
        # sample K indices with replacement per (i)
        idx1 = torch.randint(0, M, (N, K), device=est.device)
        idx2 = torch.randint(0, M, (N, K), device=est.device)
        y1 = est.gather(1, idx1.unsqueeze(-1).expand(N, K, D))  # (N,K,D)
        y2 = est.gather(1, idx2.unsqueeze(-1).expand(N, K, D))  # (N,K,D)

        s2 = (y1 - y2).abs().pow(beta).mean(dim=(0,1))        # (D,)
        es = s1 - 0.5 * s2

        if verbose:
            return torch.stack([es, s1, s2], dim=0)
        else:
            return es


def plot_2_distr(index, dpa_avgs, analogue_avgs, ds_true, w_da, vmin, vmax, title=""):
    # Plot histogram with that range
    plt.figure(figsize=(8, 5))
    
    # analogue Temperature distribution
    plt.hist(
        analogue_avgs.values,
        bins=30,
        range=(vmin, vmax),   # <-- only from -2.0 to 2.0
        density=True,
        alpha=0.75
    )
    
    # dpa generated distribution
    plt.hist(
        dpa_avgs.values,
        bins=30,
        range=(vmin, vmax),   # <-- only from -2.0 to 2.0
        density=True,
        alpha=0.75
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(title)
    
    
    # include actual value here!!
    true_temp = ds_true.TREFHT.sel(lat=slice(48,54), lon=slice(6,15)).isel(time=index).weighted(w_da).mean(dim=('lat', 'lon')).values
    plt.axvline(x=true_temp, color='red', linestyle='--', linewidth=2,
                label=f'x = {true_temp}')
    plt.show()    
    

def extract_dpa_tsamples(z500, dpa_t_samples, index, coord_ds, mask):
    z500_sample = z500[index, :] 

    # create empty index to store temperature
    temps_for_zsample = np.zeros((97, 1024))#[]
    print(temps_for_zsample.shape)
    print(type(dpa_t_samples[-1]))

    
    if isinstance(dpa_t_samples[-1], np.ndarray): # check what type arrays in dpa_t_samples list have 
        for i, sample in enumerate(dpa_t_samples):
            #print(i)
            #print(sample[index, :].shape)
            #temps_for_zsample.append(sample[index, :])
            temps_for_zsample[i,:] = restore_nan_columns(sample[index, :][None, :], mask).detach().numpy()
    
    elif isinstance(dpa_t_samples[-1], xr.DataArray): # check what type arrays in dpa_t_samples list have
        for i, sample in enumerate(dpa_t_samples):
            #print(i)
            #print(sample[index, :].shape)
            #temps_for_zsample.append(sample[index, :])
            #temps_for_zsample[i,:] = sample[index, :].values
            temps_for_zsample[i,:] = restore_nan_columns(sample[index, :].values[None, :], mask).detach().numpy()

    
    elif isinstance(dpa_t_samples[-1], torch.Tensor): # check what type arrays in dpa_t_samples list have
        for i, sample in enumerate(dpa_t_samples):
            #print(i)
            # rstore orig shape, need to add 0th dimension for that
            temps_for_zsample[i,:] = restore_nan_columns(sample[index, :][None, :], mask).detach().numpy()


    # transform temps_for_zsample into xarray
    temps_for_zsample_xr = torch_to_dataarray(temps_for_zsample, coord_ds, lat_dim=32, lon_dim=32, name="TREFHT")

    return temps_for_zsample_xr, z500_sample

def find_analogues(t_dataarray, pc_scores_ds, z500_sample, no_pcs, no_nearest):
    
    # compute differences between each sample and the "target" Z500 sample
    diff  = pc_scores_ds[:,:no_pcs] - z500_sample.numpy()[:no_pcs]

    # compute distances (euclidean norm)
    dists = np.linalg.norm(diff, axis=1)  
    
    # find the no_nearest smallest distances
    no_nearest = 97
    top_idxs = np.argsort(dists)[:no_nearest]

    # 3) slice out those rows from temperature data
    nearest_temps = t_dataarray.isel(time=top_idxs) 

    return nearest_temps, top_idxs

def load_dpa_arrays(path, mask, ds_coords, ens_members, save_path=None):
    #file_list = sorted(f for f in os.listdir(path) if f.endswith(".pt"))


    print("Now loading DPA ensemble restored including nans")
    tensor_list_raw = [torch.load(f"{path}gen{i}_te.pt", map_location=torch.device('cpu')) for i in range(1,ens_members+1)] #98 #this is loading raw arrays without nans
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
        # save to zarr
        #ds_raw.to_zarr(f"{save_path}/raw_dpa_ens_100_dataset_restored.zarr", consolidated=True, encoding={"TREFHT": {"_FillValue": None}})
        ds_raw.to_netcdf(f"{save_path}/raw_dpa_ens_100_dataset_restored.nc", format="NETCDF4")
        
    ### Restore NaNs ###
    
    # Load and stack into one tensor
    print("Now loading DPA ensemble restored including nans")
    tensor_list = [restore_nan_columns(torch.load(f"{path}gen{i}_te.pt", map_location=torch.device('cpu')), mask) for i in range(1,ens_members+1)] #98 #this is loading raw arrays without nans
    print("Tensor list length:", len(tensor_list))
    print("Tensor list elements shape:", tensor_list[0].shape)
    stacked = torch.stack(tensor_list)  # shape: (N, T, H, W)
    stacked_reshaped = stacked.reshape(stacked.shape[0], stacked.shape[1], 32, 32)


    
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
    print(ds)
    if save_path is not None:
        # save to zarr
        #ds.to_zarr(f"{save_path}/dpa_ens_100_dataset_restored.zarr", consolidated=True, encoding={"TREFHT": {"_FillValue": None}})
        ds.to_netcdf(f"{save_path}/dpa_ens_100_dataset_restored.nc", format="NETCDF4")
    

    return tensor_list, tensor_list_raw, stacked, stacked_reshaped, ds

def load_both_dpa_arrays(path, mask, ds_coords, ens_members, save_path=None, climate_list=[]):
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
        tensor_list_raw = [torch.load(f"{path}{climate}{i}_te.pt", map_location=torch.device('cpu'), weights_only=False) for i in range(1,ens_members+1)] #98 #this is loading raw arrays without nans
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
            # save to zarr
            #ds_raw.to_zarr(f"{save_path}/raw_dpa_ens_100_dataset_restored.zarr", consolidated=True, encoding={"TREFHT": {"_FillValue": None}})
            ds_raw.to_netcdf(f"{save_path}/raw_ETH_{climate}_dpa_ens_100_dataset.nc", format="NETCDF4")
            
        ### Restore NaNs ###
        
        # Load and stack into one tensor
        print("Now loading DPA ensemble restored including nans")
        tensor_list = [restore_nan_columns(torch.load(f"{path}{climate}{i}_te.pt", map_location=torch.device('cpu'), weights_only=False), mask) for i in range(1,ens_members+1)] #98 #this is loading raw arrays without nans
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
            # save to zarr
            #ds.to_zarr(f"{save_path}/dpa_ens_100_dataset_restored.zarr", consolidated=True, encoding={"TREFHT": {"_FillValue": None}})
            ds.to_netcdf(f"{save_path}/ETH_{climate}_dpa_ens_100_dataset_restored.nc", format="NETCDF4")
    

    return list_tensor_list, list_tensor_list_raw, list_stacked, list_stacked_reshaped, list_ds

def load_dpa_predicts(path, mask, save_array=False, raw_hull=(98,64000,648), restored_hull=(98,64000,1024)):
    dpa_t_samples_raw = []
    dpa_t_samples = []
    # 3D 2x3x4 tensor of zeros
    t_arr_raw = np.zeros((2,2)) # dummy 
    t_arr_restored = np.zeros((2,2)) # dummy
    if save_array:
        t_arr_raw = torch.zeros(raw_hull) # too large for memory
        t_arr_restored = torch.zeros(restored_hull) # too large for memory
    for i in range(1,98):
        print(i)
        t_sample = torch.load(f"{path}gen{i}_te.pt", map_location=torch.device('cpu'))
        dpa_t_samples_raw.append(t_sample)
        if save_array:
            dpa_t_sample_restored = restore_nan_columns(t_sample, mask)
            t_arr_raw[i, :, :] = t_sample
            t_arr_restored[i, :, :] = dpa_t_sample_restored

            
        #dpa_t_samples.append(dpa_t_sample_restored)
        
        # insert into array (causes memory problems)
        #print(dpa_t_sample_restored.shape)
        #
    return dpa_t_samples_raw, dpa_t_samples, t_arr_raw, t_arr_restored

def rank_histogram(truth, ensemble):
    """
    Compute the rank of truth among each ensemble member set.
    
    Parameters
    ----------
    truth : array_like, shape (n_cases,)
        The observed “true” values.
    ensemble : array_like, shape (n_cases, n_members)
        The ensemble forecasts for each case.
    
    Returns
    -------
    ranks : ndarray, shape (n_cases,)
        Integer ranks in 0..n_members (inclusive), where
        rank i means truth falls between sorted ensemble[i-1] and ensemble[i].
    """
    truth = np.asarray(truth)
    ens   = np.asarray(ensemble)
    n_cases, n_members = ens.shape

    # Append truth into each member set and argsort
    # For each case, count how many ensemble values are < truth:
    # that count is the rank (0 means truth below all, n_members means above all)
    ranks = np.sum(ens < truth[:, None], axis=1)
    return ranks
    
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

def plot_temperature_panel(ax, dataarray, vmax_shared, sample_nr=None, no_levels=11, levels=None, cbar=False, cmap="coolwarm", title=""):
    """
    Plot a single temperature panel on a given axis with Cartopy.

    create figure first:
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})

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
    if levels is None:
        levels = np.linspace(-5, 5, no_levels)

    else:
        levels=levels

    p = dataarray.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-vmax_shared,
        vmax=vmax_shared,
        levels=levels,
        add_colorbar=cbar
    )
    ax.set_title(title)
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
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

def plot_map(field, levels, cmap="bwr", cmap_label = ""):
    
    # create plot with a map projection
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()}, 
        figsize=(8, 6)
    )
    
    # plot the data
    field.plot(ax=ax,
               transform=ccrs.PlateCarree(),
               cmap=cmap,
               levels=levels,
               cbar_kwargs={'label': cmap_label, 'shrink': 0.7})
    
    # add coastlines and continents
    ax.coastlines(resolution="50m", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    
    # optional: gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    #plt.show()

    return fig, ax

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
    # data_np = x_tensor.detach().cpu().numpy()

    # Step 1: Convert to NumPy
    if isinstance(x_tensor, np.ndarray):
        # x is a NumPy array
        print("input array is numpy array")
        data_np = x_tensor
    elif isinstance(x_tensor, torch.Tensor):
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