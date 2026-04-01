import torch
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

import sys
sys.path.append('../')
import utils.utils as ut


def plot_multiple_dpa_time_series(true_t, 
                                  dpa_ens, 
                                  dpa_ens_mean, 
                                  true_t_fact, 
                                  dpa_ens_fact, 
                                  dpa_ens_mean_fact, 
                                  lat_min, 
                                  lat_max, 
                                  lon_min, 
                                  lon_max, 
                                  plot_year, 
                                  figsize_ts, 
                                  title_fontsize, 
                                  title, 
                                  climate):
    
    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))


    ### DPA Ensemble COUNTERFACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_std['lat']}, dims=['lat'])
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_pre = dpa_ens_mean.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_mean_ger_pre['lat']}, dims=['lat'])
    dpa_ens_mean_ger = dpa_ens_mean_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_pre = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))#.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_ger_1300_pre['lat']}, dims=['lat'])
    dpa_ens_ger_1300 = dpa_ens_ger_1300_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # counterfactual truth
    fact_truth_ger = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    

    ### DPA Ensemble FACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member")
    dpa_ens_std_ger_fact = dpa_ens_std_fact.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_fact = dpa_ens_mean_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # factual truth
    fact_truth_ger = true_t_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    
    
    # plot
    n_stds = 2
    
    # cf
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger

    # factual
    lower_env_fact = (dpa_ens_mean_ger_fact + n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    upper_env_fact = (dpa_ens_mean_ger_fact - n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    for year in plot_year:
        #year = plot_year
        lw=1.0
        times = temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values
        labels = [f'{t.month:02d}-{t.day:02d}' for t in times]

        plt.plot(range(19), temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:cyan", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:blue", linestyle="--", linewidth=lw)
        plt.plot(range(19), fact_truth_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color = "tab:orange", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:red", linestyle="--", linewidth=lw)
        
        # factual
        ax.fill_between(
            range(19), 
            lower_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:red", alpha=0.2, label=r'$\pm$' + f"2 factual DAE ensemble standard deviations",
            linewidth=0
        )

        # counterfactual
        ax.fill_between(
            range(19),
            lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:blue", alpha=0.2, label=r'$\pm$' + f"2 CF DAE ensemble standard deviations",
            linewidth=0
        )
        
        # Major ticks: every 3rd time step (with labels)
        ax.set_xticks(range(0, len(times), 3))
        
        # Minor ticks: all time steps
        ax.set_xticks(range(len(times)), minor=True)
        
        ax.set_xticklabels([lab for i, lab in enumerate(labels[::3])])


    
    ax.legend(fontsize=6, ncol=1, frameon=False)

    if len(plot_year) > 1:
        multi_year = True
    else:
        multi_year = False

    return fig, ax
    





    
    