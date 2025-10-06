import xarray as xr
import numpy as np

import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut
import dpa_ensemble as de
import importlib
import json
import torch
import matplotlib.pyplot as plt
from engression.loss_func import energy_loss, energy_loss_two_sample


def main():
    
    # load true test data
    z500_test, z500_train, mask_x_te, ds_train, ds_test, x_te_reduced, x_tr_reduced = de.load_test_data()
    
    ### time resolved e_loss ###
    e_loss_pre = torch.load("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs/train_set_ensemble_after_30_epochs/train_e_loss_time_resolved.pt")
    #print("e loss shape:", e_loss_pre.shape)

    # create xarray from e_loss
    loss_types = ["energy_loss", "S1", "S2"]

    # Create time coordinate (example: daily timestamps)
    time_coord = ds_train.time
    
    # Convert torch → numpy and wrap into xarray
    eloss_xr = xr.DataArray(
        e_loss_pre.detach().numpy(),                     # convert to NumPy
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
    aer_pre = xr.open_dataset("/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/other_data/JJA_mean_LE2_AODVIS_1850_2100.nc").AODVIS
    
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
    
    # Replace the coordinate correctly
    yearly_eloss_standardized = yearly_eloss_standardized_pre.assign_coords(
        year=("year", aodvis.time.values))

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
    #yearly_eloss_standardized.sel(loss="energy_loss").plot(label="Energy_loss")
    #europe_mean_std.plot(label="AODVIS")
    #plt.legend()
    #plt.savefig("ETH_analysis_results/dpa_train_set_ensemble/eloss_and_aerosols.png")
    #plt.show()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
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
    ax.set_title("Standardized Energy Loss vs. European AODVIS", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Standardized Value", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    fig.savefig("ETH_analysis_results/dpa_train_set_ensemble/eloss_and_aerosols.png", dpi=300)
    plt.show()

    
    #yearly_eloss_standardized.sel(loss="energy_loss").plot(label="Energy_loss")
    #yearly_eloss_standardized.sel(loss="S1").plot(label="S1")
    #yearly_eloss_standardized.sel(loss="S2").plot(label="S2")
    #plt.legend()
    #plt.savefig("ETH_analysis_results/dpa_train_set_ensemble/all_losses.png")
    #plt.show()

    # all losses
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
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
    ax.set_title("Yearly Standardized Loss Components", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Standardized Value", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Optimize layout and save
    plt.tight_layout()
    fig.savefig("ETH_analysis_results/dpa_train_set_ensemble/all_losses.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    main()

