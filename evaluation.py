import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys


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

def r2_score(y_true, y_pred, dim=0):
    """
    Compute R² per feature along `dim`.
    y_true, y_pred: torch tensors of same shape
    """
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=dim)
    ss_tot = torch.sum((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2, dim=dim)
    r2 = 1 - ss_res / ss_tot
    return r2


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

def plot_dpa_time_series(true_t, dpa_ens, dpa_ens_mean, lat_min, lat_max, lon_min, lon_max, plot_year, figsize_ts, title_fontsize, title, climate):
    
    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    #cf_temp_true_ger_pre = ds_test_1300_eth_cf.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    #cf_temp_true_ger = cf_temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    #print("cf_temp_true_ger:", cf_temp_true_ger)


    ### DPA Ensemble ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger = dpa_ens_mean.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored
    print(dpa_ens_mean_ger.shape)
    #dpa_ens_mean_ger_cf = dpa_ens_mean_cf_1300_restored.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300 = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    print("dpa_ens_ger_1300:", dpa_ens_ger_1300.values.T.shape)
    
    
    # plot
    n_stds = 2
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger

    print("dpa_ens_mean_ger:", dpa_ens_mean_ger)
    print("lower env:", lower_env)
    print("upper env:", upper_env)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    year = plot_year
    temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label=f"{climate} Truth")
    dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label=f"{climate} DPA ensmble mean")
    
    # ADD COUNTERFACTUAL TEMPERATURE HERE
    #cf_temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="CF Truth")
    #dpa_ens_mean_ger_cf.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA CF mean")
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
    ax.set_title(title, fontsize=title_fontsize)

    return fig, ax

def plot_violins(truth, dpa_ensemble, save_path):

    #test_100 = ds_test.TREFHT.isel(time=slice(-4769, 64000)) 
    # test_100 is one ensemble member
    # could be dpa_1300_fact_restored now
    test_100 = truth
    
    ## DPA ensemble
    dpa_ensmemb100 = dpa_ensemble
    
    ## DPA ensemble mean
    dpa_mean_ens_member100 = dpa_ensemble.mean(dim="ensemble_member")
    
    print("Test set:", test_100)
    print("dpa ensemble:", dpa_ensmemb100)
    print("dpa ensemble mean:", dpa_mean_ens_member100)
    
    
    ################
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15
    
    # calculate germany mean value
    ## Truth
    # test_100
    temp_true_pre_100 = test_100.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    weights_ger = np.cos(np.deg2rad(temp_true_pre_100['lat']))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_pre_100['lat']}, dims=['lat'])
    temp_true_ger = temp_true_pre_100.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    ## DPA ensemble
    # dpa_ensmemb100
    temp_dpa_ens_100_ger_pre = dpa_ensmemb100.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    temp_dpa_ens_100_ger = temp_dpa_ens_100_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    ## DPA ensemblese mean
    # dpa_mean_ens_member100
    ger_dpa_mean_ensmemb100_pre = dpa_mean_ens_member100.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    weights_ger = np.cos(np.deg2rad(ger_dpa_mean_ensmemb100_pre['lat']))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': ger_dpa_mean_ensmemb100_pre['lat']}, dims=['lat'])
    ger_dpa_mean_ensmemb100 = ger_dpa_mean_ensmemb100_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    ger_dpa_mean_ensmemb100
    
    #################

    
    truth_grouped = temp_true_ger.groupby("time.year")

    max_times = []
    min_times = []
    
    # extract indices of max/min temperatures per year from one test member (ensemble member 100)
    for key, group in truth_grouped:
        print("Year:", key)
        max_date = group.idxmax(dim="time")
        #print("group:", max_date.values)
        min_date = group.idxmin(dim="time")
        #print("group:", min_date.values)
        max_times.append((str(max_date.values)))
        min_times.append((str(min_date.values)))
    
    #################
    
    list_temp_true_max = []
    list_temp_true_min = []
    
    list_dpa_ens_max = []
    list_dpa_ens_min = []
    
    list_dpa_mean_max = []
    list_dpa_mean_min = []
    
    years = np.arange(1850, 2101, 1)
    
    for i in range(251):
        #print(i)
    
        # true (test) temperature
        list_temp_true_max.append(temp_true_ger.sel(time=max_times[i]))
        list_temp_true_min.append(temp_true_ger.sel(time=min_times[i]))
    
        # DPA ensemble
        list_dpa_ens_max.append(temp_dpa_ens_100_ger.sel(time=max_times[i]).values[:,0])
        list_dpa_ens_min.append(temp_dpa_ens_100_ger.sel(time=min_times[i]).values[:,0])
        #print(temp_dpa_ens_100_ger.sel(time=max_times[i]).values[:,0].shape)
    
        # DPA mean
        list_dpa_mean_max.append(ger_dpa_mean_ensmemb100.sel(time=max_times[i]))
        list_dpa_mean_min.append(ger_dpa_mean_ensmemb100.sel(time=min_times[i]))
     
    #################
    
    no_violins = 21
    years_dist = np.arange(0,240,20)
    n = 0
    for period in years_dist:
        print(period, period + no_violins)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
        
        # plot violin plot
        axs.violinplot(
            list_dpa_ens_max[period:period + no_violins],         # list of arrays, one array per year 
            showmeans=True,
            showmedians=False,
        )
        axs.scatter(range(1,no_violins+1), list_temp_true_max[period:period + no_violins], label="True temperatures", marker="x", color = "red")
    
        axs.set_title('DPA Maximum Temperature Distribution (Test member 100)')
        
        
        axs.yaxis.grid(True)
        
        axs.set_xticks(range(1,252,1)[:no_violins])  
        axs.set_xticklabels(years[period:period + no_violins])
        
        axs.set_xlabel('Year')
        axs.set_ylabel('T_max')
        plt.legend()
        fig.savefig(f"{save_path}/dpa_max_temp_distr{period}-{period + no_violins}.png")
        plt.show()
        n+=1

    return







    
    