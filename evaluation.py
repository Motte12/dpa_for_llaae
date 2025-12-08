import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut



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
    dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label=f"{climate} DPA ensemble\nmean")
    
    # ADD COUNTERFACTUAL TEMPERATURE HERE
    #cf_temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="CF Truth")
    #dpa_ens_mean_ger_cf.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).plot(ax=ax, label="DPA CF mean")
    # ################################## #
    
    
    # fill_between needs numpy arrays + axis
    ax.fill_between(
        dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values,   # x-axis values (datetime64)
        lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
        upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
        color="tab:orange", alpha=0.2, label=f"+/- {n_stds} DPA ensemble standard\ndeviations"
    )

    
    
    ax.legend()
    #plt.tight_layout()
    ax.set_title(title, fontsize=title_fontsize)

    return fig, ax

def create_rank_hist(test_truth, 
                     dpa_ens,
                     lat_min,
                     lat_max,
                     lon_min,
                     lon_max,
                     figsize,
                     title,
                     title_fontsize
                    ):
    
    # temp_true_ger
    temp_true_ger_pre = test_truth.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble spatial average (ensemble_member, )
    dpa_ens_ger_1300 = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    
    
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # bins from 0..n_members inclusive
    ax.hist(ranks, bins=np.arange(n_members+2) - 0.5, edgecolor="black")
    
    # set x-ticks to align with bin centers
    ax.set_xticks(np.arange(n_members + 1))
    
    # labels and title
    ax.set_xlabel("Rank of truth among ensemble members")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=title_fontsize)
    
    plt.tight_layout()
    #fig.savefig("ETH_analysis_results/ger_rank_histogram.png")
    plt.show()
    
    return fig, ax

def plot_violins(truth, 
                 dpa_ensemble, 
                 save_path,
                 lat_min,
                 lat_max,
                 lon_min,
                 lon_max,
                 in_fact_for_cf=None,
                 mode="extremes" ):
    """
    mode: whether to choose max/min ("extremes") times or random times ("random")
    """

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
    
    # calculate spatial mean value
    
    
    ## Truth

    # Test Temperatures that Max Tempersture dates are chosen from
    # test_100
    temp_true_pre_100 = test_100.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    weights_ger = np.cos(np.deg2rad(temp_true_pre_100['lat']))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_pre_100['lat']}, dims=['lat'])
    temp_true_ger = temp_true_pre_100.weighted(w_da_ger).mean(dim=('lat', 'lon'))


    # Test factual temperatures for nudged violins
    if in_fact_for_cf is not None:
        ### transient temperatures for nudged violins ###
        temp_ger_eth_trans_pre = in_fact_for_cf.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        weights_ger = np.cos(np.deg2rad(temp_ger_eth_trans_pre['lat']))
        w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_ger_eth_trans_pre['lat']}, dims=['lat'])
        temp_ger_eth_trans = temp_ger_eth_trans_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
        ##################################################
    
    ## DPA ensemble for violins
    # dpa_ensmemb100
    temp_dpa_ens_100_ger_pre = dpa_ensmemb100.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    temp_dpa_ens_100_ger = temp_dpa_ens_100_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    ## DPA ensemblese mean (not actually needed I guess?)
    # dpa_mean_ens_member100
    ger_dpa_mean_ensmemb100_pre = dpa_mean_ens_member100.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    weights_ger = np.cos(np.deg2rad(ger_dpa_mean_ensmemb100_pre['lat']))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': ger_dpa_mean_ensmemb100_pre['lat']}, dims=['lat'])
    ger_dpa_mean_ensmemb100 = ger_dpa_mean_ensmemb100_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    ger_dpa_mean_ensmemb100
    
    #################

    figure_label = "factual"
    if in_fact_for_cf is not None: 
        truth_grouped = temp_ger_eth_trans.groupby("time.year")
        figure_label = "counterfactual"
    
    else:
        truth_grouped = temp_true_ger.groupby("time.year")
        

    max_times = []
    min_times = []
    
    # extract indices of max/min temperatures per year from one test member (ensemble member 100)
    for key, group in truth_grouped:
        #print("Year:", key)
        max_date = group.idxmax(dim="time")
        # choose random timestep
        if mode == "random":
            rand_idx = np.random.randint(0, group.sizes["time"])
            print("rand idx:",rand_idx)
            print(f"Random index {rand_idx} → time {group.time[rand_idx].values}")
            max_date = group.time[rand_idx] #group.isel(time=rand_idx)
            print("max date:", max_date)
        
        #print("group:", max_date.values)
        min_date = group.idxmin(dim="time")
        #print("group:", min_date.values)
        max_times.append((str(max_date.values)))
        min_times.append((str(min_date.values)))
    
    #################
    
    list_temp_fact_max = []
    list_temp_fact_min = []
    
    list_temp_true_max = []
    list_temp_true_min = []
    
    list_dpa_ens_max = []
    list_dpa_ens_min = []
    
    list_dpa_mean_max = []
    list_dpa_mean_min = []
    
    
    years = np.arange(1850, 2101, 1)
    unique_years = np.unique(truth.time.dt.year.values) #unique years
    print("Number of years:", unique_years)
    print("Iteration:", (np.arange(unique_years[0],unique_years[-1])-1850))
    print("Max times:", max_times)
    print("temp_true_ger", temp_true_ger.time)
    #for i in (np.arange(unique_years[0],unique_years[-1])-1850):
    for i in (range(unique_years.shape[0])):
        print(i)
        print(max_times[i])

        # for cf
        if in_fact_for_cf is not None:
            list_temp_fact_max.append(temp_ger_eth_trans.sel(time=max_times[i]))
            list_temp_fact_min.append(temp_ger_eth_trans.sel(time=min_times[i]))
    
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
    
    no_violins = 20 #21

    # use this if its not working any more
    #years_dist = np.arange(unique_years[0],unique_years[-1],no_violins)-1850 #np.arange(0,240,20)
    years_dist = np.arange(0, unique_years[-1]-unique_years[0]+1, no_violins)
    
    print(years_dist)

    # create large figure
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(40, 25))
    print(axes)
    axs = axes.flatten()
    print(axs)
    print("list_dpa_ens_max", len(list_dpa_ens_max))
    #print("list_dpa_ens_max", list_dpa_ens_max)
    n = 0
    for period in years_dist:
        print("period, period + no_violins:", int(n*no_violins), int(n*no_violins) + no_violins)
        #fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
        print("n:",n)
        # plot violin plot
        if True:
            axs[n].violinplot(
                list_dpa_ens_max[period:period + no_violins],         # list of arrays, one array per year 
                showmeans=True,
                showmedians=False,
            )

        if False:
            axs[n].violinplot(
                list_dpa_ens_max[int(n*no_violins):int(n*no_violins) + no_violins],         # list of arrays, one array per year 
                showmeans=True,
                showmedians=False,
            )
        # set number of violins
        no_violins = len(list_dpa_ens_max[period:period + no_violins])
        true_color = "red"
        true_label = "Test factual max Temperature"
        if in_fact_for_cf is not None:
            axs[n].scatter(range(1,no_violins+1), list_temp_fact_max[period:period + no_violins], label="Test factual max temperatures", marker="x", color = "red")
            true_color = "orange"
            true_label = "Nudged Test Temperature"
        #axs[n].scatter(range(1,no_violins+1), list_temp_true_max[period:period + no_violins], label=true_label, marker="x", color = true_color)
        axs[n].scatter(range(1,no_violins+1), list_temp_true_max[period:period + no_violins], label=true_label, marker="x", color = true_color)
    
        axs[n].set_title('DPA Maximum Temperature Distribution (Test member 1300)')
        
        
        axs[n].yaxis.grid(True)
        
        axs[n].set_xticks(range(1,252,1)[:no_violins:5])  
        axs[n].set_xticklabels(years[period:period + no_violins:5])
        
        axs[n].set_xlabel('Year')
        axs[n].set_ylabel('T_max')
        # create a proxy handle for the legend
        vio_patch = mpatches.Patch(color="lightblue", label="DPA Ensemble")
        axs[n].legend(handles=[vio_patch])
        axs[n].legend()
        #fig.savefig(f"{save_path}/dpa_max_temp_distr{period}-{period + no_violins}.png")
        #plt.show()
        n+=1

        if n == 12:
            break
    
    fig.savefig(f"{save_path}/{figure_label}_{mode}_dpa_max_temp_distr.png")
    plt.show()
    return fig

def compute_autocorrelation(x, mask, ds_coords):
    """
    x: 2d input torch array of shape (timesteps, features)

    returns: autocorr, autocorr_xr
    """
    if x.ndim == 2:
        data_centered = x - x.mean(dim=0, keepdim=True) # center data by subtracting mean along time
        
        # compute Lag = 1 autocorrelation
        numerator = (data_centered[:-1, :] * data_centered[1:, :]).sum(dim=0)
        denominator = (data_centered ** 2).sum(dim=0)
        autocorr = numerator / denominator
    
        # turn into xarray
        autocorr_restored = ut.restore_nan_columns(autocorr, mask)
        autocorr_xr = ut.torch_to_dataarray(autocorr_restored, ds_coords)

    elif x.ndim > 2:
        # center
        test_data_centered_mult = x - x.mean(dim=1, keepdim=True) # dim1=time
    
        # Lag = 1 autocorrelation
        # test_data_centered_mult shape = (ensemble members: 100, time: 969, features: 648)
        numerator_mult = (test_data_centered_mult[:, :-1, :] * test_data_centered_mult[:, 1:, :]).sum(dim=1)
        denominator_mult = (test_data_centered_mult ** 2).sum(dim=1)
        autocorr = numerator_mult / denominator_mult

        # add empty xarray
        autocorr_xr = ['no xarray']
        
    
    return autocorr, autocorr_xr
    





    
    