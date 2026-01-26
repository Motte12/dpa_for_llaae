import xarray as xr
import json

import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/data_exploration')
import dpa_ensemble as de
import utils as ut
import pytorch_quantile_regression as pqr
import evaluation
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    
    # Factual
    # load true test data
    settings_file_path = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v4_dpa_train_settings.json"
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
        
    ### Load temperature data ###
    ds_test_eth_fact = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    
    # load DPA ensemble
    dpa_eth_fact = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_gen_dpa_ens_20_dataset_restored.nc")

    # Counterfactual
    # load true test data
    settings_file_path = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v4_dpa_train_settings.json"
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
        
    ### Load temperature data ###
    ds_test_eth_cf = xr.open_dataset(settings['dataset_trefht_eth_nudged_shifted'])
    
    # load DPA ensemble
    dpa_eth_cf = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/v4_model/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128_bnisTrue/dpa_ensemble_after_20_epochs/eth_ensemble_after_20_epochs/ETH_cf_gen_dpa_ens_20_dataset_restored.nc")

    # compute domain average temperature
    dpa_eth_ens_spatial_mean = dpa_eth_fact.TREFHT.mean(dim=("lat","lon"))
    dpa_eth_ens_spatial_mean_cf = dpa_eth_cf.TREFHT.mean(dim=("lat","lon"))
    
        
    fact_dpa_mean = dpa_eth_fact.TREFHT.mean(dim="ensemble_member")    
    cf_dpa_mean = dpa_eth_cf.TREFHT.mean(dim="ensemble_member")

    ######################
    ### Load test data ###
    ######################
    # load my data
    settings_file_path = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v4_dpa_train_settings.json"
    
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
    
    print("Top-level keys:", list(settings.keys()))
    
    # Load LE temperature data
    # Train data
    trefht_le = xr.open_dataset(settings['dataset_trefht'])
    
    # Load LE Z500 
    z500_le = xr.open_dataset(settings["dataset_z500"])
    predictors_combined_le = z500_le.pseudo_pcs.values
    
    # Load ETH ensemble data
    # Test data
    trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    
    z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
    
    
    # concatenate data 
    predictors_combined_eth = z500_eth.pseudo_pcs #xr.concat([gmt_eth_expanded, z500_eth.pseudo_pcs], dim="mode")
    predictors_combined_eth
    # germany domain 
    ### Germany ###
        
    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15
    
    # cut train data
    trefht_le_ger = trefht_le.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    print(trefht_le_ger)
    
    # cut test data
    trefht_eth_ger = trefht_eth.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    print(trefht_eth_ger)
    
    # calculate weighted means
    #weights
    weights_ger_pre = np.cos(np.deg2rad(trefht_le_ger["lat"]))
    weights_ger = weights_ger_pre / weights_ger_pre.sum()
    
    # training data
    trefht_le_ger_mean = trefht_le_ger.TREFHT.weighted(weights_ger).mean(dim=("lat", "lon"))
    
    # test_data
    trefht_eth_ger_mean = trefht_eth_ger.TREFHT.weighted(weights_ger).mean(dim=("lat", "lon"))
    
    print("Predictors train shape:", predictors_combined_le.shape)
    print("Predictands train shape:", trefht_le_ger_mean.shape)
    print("#######")
    print("Predictors train shape:", predictors_combined_eth.shape)
    print("Predictands test shape:", trefht_eth_ger_mean.shape)

    # assign test simulation
    trefht_eth_ger_mean_1300 = trefht_eth_ger_mean.isel(time=slice(0,4769))

    ### DPA Ensemble ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_eth_fact.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).std(dim="ensemble_member") 
    dpa_ens_std_ger = dpa_ens_std.weighted(weights_ger).mean(dim=('lat', 'lon')).isel(time=slice(0,4769))
    
    # ensemble mean, germany mean 
    dpa_ens_mean_ger = fact_dpa_mean.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(weights_ger).mean(dim=('lat', 'lon')).isel(time=slice(0,4769)) # before: dpa_ensemble_restored
    print(dpa_ens_mean_ger.shape)

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300 = dpa_eth_fact.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max)).weighted(weights_ger).mean(dim=('lat', 'lon')).isel(time=slice(0,4769))
    dpa_ens_ger_1300

    # dpa ensemble germany mean
    dpa_ens_ger = dpa_eth_fact.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    dpa_ens_ger_mean = dpa_ens_ger.weighted(weights_ger).mean(dim=('lat', 'lon')).isel(time=slice(0,4769))

    ####################
    ### Linear Model ###
    ####################
    
    # standardize predictors
    ###
    ## train
    train_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
    X_torch = torch.from_numpy(train_predictors)
    y_torch = torch.from_numpy(trefht_le_ger_mean.values)[:90*4769]
    
    ## validation
    validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :])
    X_val_torch = torch.from_numpy(validation_predictors)
    y_val_torch = torch.from_numpy(trefht_le_ger_mean.values)[90*4769:]
    
    ## test
    test_predictors, _, _ = ut.standardize_numpy(z500_eth.pseudo_pcs.values)
    x_test_torch = torch.from_numpy(test_predictors)
    
    print("X:", X_torch.shape)
    print("y:", y_torch.shape)
    
    print("X_val:", X_val_torch.shape)
    print("y_val:", y_val_torch.shape)
    
    print("X test:", x_test_torch.shape)
    
    # fit linear model
    
    #X = X_torch[:4769, :10].float()
    X = torch.cat(
        [X_torch[:, :20], X_torch[:, -1:].clone()],
        dim=1
    )
    y = y_torch[:].float()
    
    # Add bias column
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    
    # Solve min ||X w - y||
    w = torch.linalg.lstsq(X_aug, y).solution
    
    weights = w[:-1]
    bias = w[-1]
    
    
    # apply model to test data
    test_predictors = torch.cat(
        [x_test_torch[:, :20], x_test_torch[:, -1:].clone()],
        dim=1
    )
    #y_pred_test = x_test_torch[:,:10].float() @ weights + bias
    y_pred_test = test_predictors.float() @ weights + bias
    print(type(y_pred_test))
    # evaluation
    mse = torch.mean((y_pred_test - torch.from_numpy(trefht_eth_ger_mean.values)) ** 2)
    rmse = torch.sqrt(mse)
    print("Test RMSE:", rmse.item())
    
    # create xarray from predictions
    
    y_pred_test_xr = xr.DataArray(y_pred_test.detach().numpy(),
                                  dims = ["time"],
                                  coords = {"time": trefht_eth_ger_mean.time},
                                  name="TREFHT")

    # load absolute yearly maximum values
    idx_abs = np.load("yearly_absolute_max_indices_ger.npy")
    plot_years = np.arange(1850,2101)
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- main time series ---
    
    # plot true temperature anomalies 
    plt.plot(
        plot_years,
        trefht_eth_ger_mean_1300.values[idx_abs], #da_max_year.values[idx_abs],
        marker='x',
        label = "truth"
    )
    
    # plot linear model
    plt.plot(
        plot_years,
        y_pred_test_xr.isel(time=slice(0,4769)).isel(time=idx_abs).values,
        marker='x',
        color="grey",
        label="linear model (x: 10 PCs + GMT)"
    )
    
    
    # plot DPA ensemble mean
    plt.plot(
        plot_years,
        dpa_ens_mean_ger.isel(time=idx_abs).values,
        marker='o',
        label = "DPA mean"
    )

    n_stds = 2
    upper_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger.TREFHT 
    lower_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger.TREFHT
    
    # fill_between
    plt.fill_between(
        plot_years,   # x-axis values (datetime64)
        lower_env.isel(time=idx_abs).values,        # lower bound
        upper_env.isel(time=idx_abs).values,        # upper bound
        color="tab:orange", alpha=0.2, label=f"+/- {n_stds} DPA ensemble standard\ndeviations"
    )
    
    
    # quantile predictions with quantile interval (0.05-0.95)
    #ax.plot(da_max_year_pre.year.values, quantile_predictions[0:4769,9][idx_abs], label = 'quantile regression:\npredicted median', color='k')
    #ax.fill_between(
    #    da_max_year_pre.year.values,   # x-axis values (datetime64)
    #    quantile_predictions[0:4769,0][idx_abs],        # lower bound
    #    quantile_predictions[0:4769,18][idx_abs],        # upper bound
    #    color="grey", alpha=0.2, label=f"quantile regression:\n0.05 - 0.95 quantile range"
    #)
    
    ax.set_xlabel("Year")
    ax.set_ylabel(trefht_eth_ger_mean.name or "Value")
    ax.set_title("Germany Annual Maximum Temperatures")
    ax.set_xlim(1950, 2000)
    ax.legend(loc="upper left")
    
    # --- create a separate marginal axis on the RIGHT ---
    divider = make_axes_locatable(ax)
    #ax_dist = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)
    
    divider = make_axes_locatable(ax)
    ax_dist = divider.append_axes(
        "right",
        size="20%",
        pad=0.3,   # ← increase this value to add more space
        sharey=ax
    )

    
    
    # --- data ---
    truth_vals = trefht_eth_ger_mean_1300.values[idx_abs] #da_max_year.values
    model_vals = y_pred_test_xr.isel(time=slice(0,4769)).isel(time=idx_abs).values #linear model
    dpa_vals = dpa_ens_ger_mean.TREFHT.values[0,idx_abs].flatten() # plot one ensemble member or all 100 members
    
    
    y_grid = np.linspace(
        min(truth_vals.min(), model_vals.min()),
        max(truth_vals.max(), model_vals.max()),
        400
    )
    
    # KDEs
    truth_kde = gaussian_kde(truth_vals)
    dpa_kde = gaussian_kde(dpa_vals)
    model_kde = gaussian_kde(model_vals)
    
    # Histograms (horizontal)
    ax_dist.hist(
        truth_vals,
        bins=20,
        orientation="horizontal",
        density=True,
        alpha=0.3,
        color="tab:blue"
    )
    
    ax_dist.hist(
        dpa_vals,
        bins=20,
        orientation="horizontal",
        density=True,
        alpha=0.3,
        color="tab:orange"
    )
    
    ax_dist.hist(
        model_vals,
        bins=20,
        orientation="horizontal",
        density=True,
        alpha=0.3,
        color="tab:green"
    )
    
    # KDE lines
    ax_dist.plot(truth_kde(y_grid), y_grid, color="tab:blue")
    ax_dist.plot(dpa_kde(y_grid), y_grid, color="tab:orange")
    ax_dist.plot(model_kde(y_grid), y_grid, color="tab:green")
    
    
    # --- styling: marginal only ---
    ax_dist.set_xlabel("Density")
    ax_dist.yaxis.set_visible(False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.ylim(-2,10)
    plt.savefig("figures/test_figure.pdf")
    #plt.show()

    if True:

        ### Load quantile regression model
        # show quantile regression model 
        # ---- Load metadata ----
        model_path = "/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v4_data_quantile_regression_ger_gradient_descent_2025-12-08_14-39/"
        with open(f"{model_path}metadata.json", "r") as f:
            meta = json.load(f)
        
        quantiles = meta["quantiles"]
        n_features = meta["n_features"]
        n_quantiles = len(quantiles)
        
        # ---- Load checkpoint ----
        ckpt_path = meta["last_checkpoint"]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # ---- Rebuild model ----
        model = pqr.LinearMultiQuantileRegressor(
            n_features=n_features,
            n_quantiles=n_quantiles
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # ---- Move to GPU (optional) ----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # ---- Prepare test data ----
        # load my data
        settings_file_path = f"../joint_training/v4_dpa_train_settings.json" #used v2 here for a long time
        
        with open(settings_file_path, 'r') as file:
                settings = json.load(file)
        
        # Load Z500 data
        z500_test = xr.open_dataset(settings['dataset_z500_eth_test']).pseudo_pcs
        
        print("v1 data, still standardized here")
        z500_test_np, _, _ = ut.standardize_numpy(z500_test.values)
        X_test_torch = torch.from_numpy(z500_test_np.astype("float32")).to(device)
        print(X_test_torch.shape)
        
        if False:
            X_test_torch[:,-1] = -0.7389813694652794
        # ---- Predict from Qu. regression model ----
        with torch.no_grad():
            preds = model(X_test_torch)   # shape (N_test, n_quantiles)
        quantile_predictions = preds.cpu().numpy()
        
        print(preds.shape)
        quantile_predictions = preds.cpu().numpy()
        print("quantile predictions:", quantile_predictions)
        
        # Temperature Test data
        if False:
            trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_nudged_shifted'])
        else:
            trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
        print(trefht_eth)
        
        print("##############")
        print("###Datasets###")
        print("##############")
        print(xr.open_dataset(settings['dataset_trefht_eth_nudged_shifted']))
        print(xr.open_dataset(settings['dataset_trefht_eth_transient']))

        ### Germany ###
    
        # coordinates 
        ger_lat_min = 48
        ger_lat_max = 54
        ger_lon_min = 6
        ger_lon_max = 15
        quantiles = meta["quantiles"]

        

        
        ############
        ### Plot ###
        ############

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # --- main time series ---
        
        # true test temperature series at yearly max absolute temperatures (idx_abs)
        ax.plot(
            plot_years,
            trefht_eth_ger_mean_1300.values[idx_abs],
            marker="x",
            label="truth"
        )
        
        # linearly predicted temperatures
        ax.plot(
            plot_years,
            y_pred_test_xr.isel(time=slice(0,4769)).isel(time=idx_abs),
            marker="x",
            label="linear model (x: 10 PCs + GMT)"
        )
        
        # quantile predictions with quantile interval (0.05-0.95)
        ax.plot(plot_years, quantile_predictions[0:4769,9][idx_abs], label = 'quantile regression:\npredicted median', color='k')
        ax.fill_between(
            plot_years,   # x-axis values (datetime64)
            quantile_predictions[0:4769,0][idx_abs],        # lower bound
            quantile_predictions[0:4769,18][idx_abs],        # upper bound
            color="grey", alpha=0.2, label=f"quantile regression:\n0.05 - 0.95 quantile range"
        )
        
        ax.set_xlabel("Year")
        ax.set_ylabel(trefht_eth_ger_mean.name or "Value")
        ax.set_title("Germany Annual Maximum Temperatures")
        ax.set_xlim(1950, 2000)
        ax.legend(loc="upper left")
        
        # --- create a separate marginal axis on the RIGHT ---
        divider = make_axes_locatable(ax)
        #ax_dist = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)
        
        divider = make_axes_locatable(ax)
        ax_dist = divider.append_axes(
            "right",
            size="20%",
            pad=0.3,   # ← increase this value to add more space
            sharey=ax
        )
        
        # --- data ---
        truth_vals = trefht_eth_ger_mean_1300.values[idx_abs] #da_max_year.values
        model_vals = y_pred_test_xr.isel(time=slice(0,4769)).isel(time=idx_abs).values
        
        y_grid = np.linspace(
            min(truth_vals.min(), model_vals.min()),
            max(truth_vals.max(), model_vals.max()),
            400
        )
        
        # KDEs
        truth_kde = gaussian_kde(truth_vals)
        model_kde = gaussian_kde(model_vals)
        
        # Histograms (horizontal)
        ax_dist.hist(
            truth_vals,
            bins=20,
            orientation="horizontal",
            density=True,
            alpha=0.3,
            color="tab:blue"
        )
        
        ax_dist.hist(
            model_vals,
            bins=20,
            orientation="horizontal",
            density=True,
            alpha=0.3,
            color="tab:orange"
        )
        
        # KDE lines
        ax_dist.plot(truth_kde(y_grid), y_grid, color="tab:blue")
        ax_dist.plot(model_kde(y_grid), y_grid, color="tab:orange")
        
        # --- styling: marginal only ---
        ax_dist.set_xlabel("Density")
        ax_dist.yaxis.set_visible(False)
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
        
        plt.tight_layout()
        plt.ylim(-2,10)
        plt.savefig("figures/test_figure_motivation.pdf")
        plt.show()


if __name__ == "__main__":
    main()

