import xarray as xr
import numpy as np
import json
import matplotlib.pyplot as plt
import cftime
import argparse
from joblib import Parallel, delayed
import multiprocessing
from cftime import DatetimeNoLeap
from datetime import datetime

def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pcs", type=int, default=10)
    parser.add_argument("--n_analogues", type=int, default=5)
    parser.add_argument("--settings_file_path", type=str, default = "../joint_training/v2_dpa_train_settings.json")
    parser.add_argument("--analogue_ens_size", type=int, default=100)
    parser.add_argument("--anal_ens_save_path", type=str, default = "/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/analogues/")
    args = parser.parse_args()

    no_pcs = args.no_pcs
    n_analogues = args.n_analogues
    
    #################
    ### Load Data ###
    #################
    # load my data
    settings_file_path = args.settings_file_path
    
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)

    ################################
    ### Load LE temperature data ###
    ################################
    # Train data
    trefht_le = xr.open_dataset(settings['dataset_trefht'])
    
    # Load LE Z500 
    z500_le = xr.open_dataset(settings["dataset_z500"])
    
    # load LE GMT
    gmt_le_pre = xr.open_dataset(settings["GMT_LE"])
    gmt_le = (gmt_le_pre - gmt_le_pre.mean()) / gmt_le_pre.std()
    
    # concatenate data 
    gmt_le_expanded = gmt_le.TREFHT.expand_dims(mode=[-1]).T  # add 'mode' dimension with length 1
    predictors_combined_le = xr.concat([gmt_le_expanded, z500_le.pseudo_pcs], dim="mode")
    
    ##############################
    ### Load ETH ensemble data ###
    ##############################
    # Test data
    trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    
    z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
    
    gmt_eth_pre = xr.open_dataset(settings["gmt_eth"])
    gmt_eth = (gmt_eth_pre - gmt_eth_pre.mean()) / gmt_eth_pre.std()
    
    # concatenate data 
    gmt_eth_expanded = gmt_eth.TREFHT.expand_dims(mode=[-1]).T  # add 'mode' dimension with length 1
    predictors_combined_eth = xr.concat([gmt_eth_expanded, z500_eth.pseudo_pcs], dim="mode")

    ########################
    ### Create Analogues ###
    ########################

    # ----------------------------------------------------
    # Settings
    # ----------------------------------------------------
    # follwin defined above
    #no_pcs = 10           # number of principal components used
    #n_analogues = 5       # number of analogues to retain per ETH timestep
    
    # ----------------------------------------------------
    # Shortcuts to PC arrays
    # ----------------------------------------------------
    # LE PCs (training / analogue pool), restrict to first no_pcs
    pcs_le = z500_le.pseudo_pcs.isel(mode=slice(0, no_pcs))       # (time_le, mode)
    pcs_eth = z500_eth.pseudo_pcs.isel(mode=slice(0, no_pcs))     # (time_eth, mode)
    
    time_le = pcs_le.time
    time_eth = pcs_eth.time
    
    # Assume trefht_le and trefht_eth on same lat/lon grid
    lat = trefht_le.lat
    lon = trefht_le.lon
    
    n_eth = 20
    #n_eth = time_eth.size
    #if n_eth != 14307:
    #    n_eth = 14307
    #    print("n_eth set to 14307 manually")
    time_eth = time_eth[:n_eth]
    
    # ----------------------------------------------------
    # Time range for LE pool (for clipping the ±15-year window)
    # ----------------------------------------------------
    tmin_le = time_le.min().item()   # earliest LE time (cftime.DatetimeNoLeap)
    tmax_le = time_le.max().item()   # latest LE time
    
    print("LE time range:", tmin_le, "to", tmax_le)
    
    # ----------------------------------------------------
    # Allocate arrays for analogue fields and times
    # ----------------------------------------------------
    # analogue temperatures: (time_eth, analogue, lat, lon)
    analogue_temps_all = np.empty((n_eth, n_analogues, lat.size, lon.size),
                                  dtype=np.float32)
    
    # store which LE times were picked as analogues
    analogue_times_all = np.empty((n_eth, n_analogues), dtype=object)
    
    # ----------------------------------------------------
    # Main loop over all ETH timesteps
    # ----------------------------------------------------
    for i in range(n_eth):
        print(i)
        # 1) PC scores of current ETH sample
        ana_sample = pcs_eth.isel(time=i)     # (mode)
        
        # 2) Euclidean distance in PC space to all LE samples
        diff = (((pcs_le - ana_sample)**2).sum(dim="mode"))**0.5   # (time_le,)
    
        # 3) Define desired ±15-year window around this ETH date
        t_eth = ana_sample.time
        #year  = int(t_eth.dt.year)
        #month = int(t_eth.dt.month)
        #day   = int(t_eth.dt.day)
    
        #desired_start = cftime.DatetimeNoLeap(year - 15, month, day)
        #desired_end   = cftime.DatetimeNoLeap(year + 15, month, day)
    
        # 4) Clip to available LE time range [tmin_le, tmax_le]
        #start = max(desired_start, tmin_le)
        #end   = min(desired_end,   tmax_le)
    
        # subset diff to this clipped window
        #diff_sub = diff.sel(time=slice(start, end))
    
        ###
        # slice without using sel
        month = ana_sample.time.dt.month.values
        day = ana_sample.time.dt.day.values
        
        start = cftime.DatetimeNoLeap((ana_sample.time.dt.year.values-15),month,day)
        end   = cftime.DatetimeNoLeap((ana_sample.time.dt.year.values+15),month,day)
        
        mask = (diff.time >= start) & (diff.time <= end)
        diff_sub = diff.isel(time=mask)
        diff_sub
        ###
    
        if diff_sub.time.size < n_analogues:
            raise ValueError(
                f"Not enough analogue candidates for ETH index {i}: "
                f"only {diff_sub.time.size} within [{start}, {end}]"
            )
    
        # 5) Find indices of n_analogues smallest distances
        vals = diff_sub.values
        idx_sub = np.argpartition(vals, n_analogues)[:n_analogues]
        # sort those n_analogues by actual distance
        order = np.argsort(vals[idx_sub])
        idx_sub_sorted = idx_sub[order]
        print(idx_sub_sorted)
    
        # mask trefht_le
        trefht_le_sub = trefht_le.isel(time=mask)
    
        # 6) Corresponding LE times and temperature fields
        ana_times = diff_sub.time.isel(time=idx_sub_sorted)
        ana_temp_fields = trefht_le_sub.TREFHT.isel(time=idx_sub_sorted)     # (analogue, lat, lon)
    
        # 7) Store in arrays
        analogue_temps_all[i, :, :, :] = ana_temp_fields.values.transpose(2, 1, 0)
        analogue_times_all[i, :] = ana_times.values
    
        if i % 100 == 0:
            print(f"Processed ETH time index {i+1}/{n_eth}")



    ###############################
    ### Create dataset and save ###
    ###############################

    # ----------------------------------------------------
    # Wrap results in xarray DataArray / Dataset
    # ----------------------------------------------------
    analogue_temps_da = xr.DataArray(
        analogue_temps_all.transpose(0,1,3,2),
        dims=("time", "analogue", "lat" , "lon"),
        coords={
            "time": time_eth,
            "analogue": np.arange(n_analogues),
            "lat": trefht_eth.lat,
            "lon": trefht_eth.lon,
        },
        name="analogue_temp"
    )
    #print(analogue_temps_da)
    
    analogue_times_da = xr.DataArray(
        analogue_times_all,
        dims=("time", "analogue"),
        coords={
            "time": time_eth,
            "analogue": np.arange(n_analogues),
        },
        name="analogue_time"
    )
    #print(analogue_times_da)
    
    true_temp_da = trefht_eth.TREFHT.isel(time=slice(0,n_eth))  # (time_eth, lat, lon)
    #print(true_temp_da)
    ds_analogues = xr.Dataset(
        {
            "true_temp": true_temp_da,          # ETH temperature field
            "analogue_temp": analogue_temps_da, # analogue LE temps
            "analogue_time": analogue_times_da  # LE times of analogues
        }
    )

    # create ensemble from analogues
    ds = ds_analogues  # just shorthand
    n_time = ds.dims["time"]
    n_analogues = ds.dims["analogue"]   # should be 5
    n_ens = args.analogue_ens_size                         # desired ensemble size
    
    # 1) Draw random analogue indices for each (time_eth, ensemble)
    rng = np.random.default_rng(seed=42)  # optional seed for reproducibility
    choice = rng.integers(0, n_analogues, size=(n_time, n_ens))
    
    choice_da = xr.DataArray(
        choice,
        dims=("time", "analogue_ensemble_member"),
        coords={
            "time": ds["time"],
            "analogue_ensemble_member": np.arange(n_ens),
        },
        name="analogue_choice_index"
    )
    
    # 2) Use advanced xarray indexing to select analogue_temp accordingly
    # This gives shape: (time_eth, ensemble, lat, lon)
    sampled_temp = ds["analogue_temp"].isel(analogue=choice_da)
    
    sampled_temp.name = "ensemble_temp"
    
    # 3) Build new dataset: copy original + add ensemble dimension + ensemble temps
    ds_ens = ds.copy()
    
    # Add the ensemble coordinate (variables without ensemble dim are unchanged)
    #ds_ens = ds_ens.assign_coords(ensemble=("ensemble", np.arange(n_ens)))
    
    # Add the new variable with ensemble dim
    ds_ens["ensemble_temp"] = sampled_temp

    # Convert the time coordinates
    def convert_cftime_to_datetime(ds):
        # Assuming 'time' is the name of your time coordinate
        if 'time' in ds.coords:
            # Convert each time value in the dataset
            ds['time'] = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                          for t in ds['time'].values]
        return ds
    
    # Example: Convert time coordinates in your dataset
    #ds_ens = convert_cftime_to_datetime(ds_ens)
    
    #ds_ens["time"] = np.arange(14307)
    ds_ens.to_netcdf(f"{args.anal_ens_save_path}analogue_ensemble_{no_pcs}pcs_{n_analogues}analogues_{n_ens}analoguemembers_sub20.nc")
    print(ds_ens)
    print("Analogue ensemble saved")
    # Check if any time values are None or NaT

    




    
    




if __name__ == "__main__":
    main()