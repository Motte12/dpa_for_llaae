import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import xarray as xr
import json
from joblib import Parallel, delayed
import multiprocessing
import os

# load my data
settings_file_path = "../joint_training/v2_dpa_train_settings.json"

with open(settings_file_path, 'r') as file:
        settings = json.load(file)

print("Top-level keys:", list(settings.keys()))

# Load LE temperature data
# Train data
trefht_le = xr.open_dataset(settings['dataset_trefht'])
print(trefht_le)

# Load LE Z500 
z500_le = xr.open_dataset(settings["dataset_z500"])
print(z500_le)

# load LE GMT
gmt_le_pre = xr.open_dataset(settings["GMT_LE"])
gmt_le = (gmt_le_pre - gmt_le_pre.mean()) / gmt_le_pre.std()
print(gmt_le)

# concatenate data 
gmt_le_expanded = gmt_le.TREFHT.expand_dims(mode=[-1]).T  # add 'mode' dimension with length 1
predictors_combined_le = xr.concat([gmt_le_expanded, z500_le.pseudo_pcs], dim="mode")
predictors_combined_le
#gmt_le_expanded

# Load ETH ensemble data
# Test data
trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
print(trefht_eth)

z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
print(z500_eth)

gmt_eth_pre = xr.open_dataset(settings["gmt_eth"])
gmt_eth = (gmt_eth_pre - gmt_eth_pre.mean()) / gmt_eth_pre.std()

# concatenate data 
gmt_eth_expanded = gmt_eth.TREFHT.expand_dims(mode=[-1]).T  # add 'mode' dimension with length 1
predictors_combined_eth = xr.concat([gmt_eth_expanded, z500_eth.pseudo_pcs], dim="mode")
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
trefht_le_ger_mean = trefht_le_ger.weighted(weights_ger).mean(dim=("lat", "lon"))
trefht_le_ger_mean

# test_data
trefht_eth_ger_mean = trefht_eth_ger.weighted(weights_ger).mean(dim=("lat", "lon"))
trefht_eth_ger_mean

# do for several quantiles
quantiles = np.linspace(0.05, 0.95, 19)  # e.g. 5% to 95%
print(quantiles)
models = {}

no_samples = 476900
no_predictors = 21#1001

X = predictors_combined_le.values[:no_samples, :no_predictors] #ds[predictors].to_array().T.values  # shape (n_samples, n_features)
y = trefht_le_ger_mean.TREFHT.values[:no_samples]#ds["y"].values

#for q in quantiles:
#    print("Quantile:", q)
#    model = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
#    models[q] = model

def fit_quantile_model(q, X, y, alpha=0.0, solver="highs"):
    """Fit a QuantileRegressor for a single quantile."""
    print(f"Fitting quantile {q:.2f}")
    model = QuantileRegressor(quantile=q, alpha=alpha, solver=solver)
    model.fit(X, y)
    return q, model

# number of workers: use all available cores by default
n_jobs = multiprocessing.cpu_count()  # or set manually, e.g. n_jobs = 8

results = Parallel(n_jobs=n_jobs)(
    delayed(fit_quantile_model)(q, X, y, alpha=0.0, solver="highs")
    for q in quantiles
)

# rebuild the dict {quantile: model}
models = {q: model for q, model in results}


# save results
# -------------------------------
# 1. Predict quantiles for training data
# -------------------------------
# For each fitted model, predict conditional quantiles for the training predictors
y_pred_matrix = np.column_stack([models[q].predict(X) for q in quantiles])  # shape: (n_samples, n_quantiles)

print("Predictions shape:", y_pred_matrix.shape)

# -------------------------------
# 2. Create xarray Dataset for actual + predicted values
# -------------------------------
ds_results = xr.Dataset(
    {
        "y_actual": (["sample"], y),
        "y_pred":   (["sample", "quantile"], y_pred_matrix),
    },
    coords={
        "sample": np.arange(len(y)),
        "quantile": quantiles,
    },
    attrs={
        "description": "Actual vs. predicted values from quantile regression models",
        "n_samples": len(y),
        "n_predictors": no_predictors,
        "quantiles": f"{quantiles[0]}–{quantiles[-1]}",
    }
)

print(ds_results)

# -------------------------------
# 3. Create xarray Dataset for model parameters (coefs + intercepts)
# -------------------------------
coefs = np.stack([models[q].coef_ for q in quantiles])           # shape (n_quantiles, n_predictors)
intercepts = np.array([models[q].intercept_ for q in quantiles])  # shape (n_quantiles,)

ds_models = xr.Dataset(
    {
        "coefficients": (["quantile", "predictor"], coefs),
        "intercepts":   (["quantile"], intercepts),
    },
    coords={
        "quantile": quantiles,
        "predictor": np.arange(no_predictors),
    },
    attrs={
        "description": "Fitted quantile regression model parameters",
        "alpha": 0.0,
        "solver": "highs",
    }
)

print(ds_models)

# -------------------------------
# 4. Save both datasets to NetCDF
# -------------------------------
output_dir = "/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/ger_mean_21_predictors_all_samples_lasso_0.0"
os.makedirs(output_dir, exist_ok=True)

ds_results.to_netcdf(os.path.join(output_dir, "quantile_regression_predictions_1st_21_predictors_all_data.nc"))
ds_models.to_netcdf(os.path.join(output_dir, "quantile_regression_models_1st_21_predictors_all_data.nc"))

print(f"✅ Saved prediction results and model parameters to: {output_dir}")

# -------------------------------
# Q - Q plot and save
# -------------------------------
#ds_results = ger_qu_regr_predictions
#ds_models = ger_qu_regr_models

print(ds_results)
print(ds_models)

# Extract arrays
y_true = ds_results["y_actual"].values
y_pred = ds_results["y_pred"].values  # shape (n_samples, n_quantiles)
quantiles = ds_results["quantile"].values

# -------------------------------
# 2. Compute empirical quantiles of true data
# -------------------------------
# Empirical quantile of the actual data (unconditional)
y_empirical = np.quantile(y_true, quantiles)

# Mean predicted quantile over all samples (conditional median shape)
y_pred_mean = y_pred.mean(axis=0)

# -------------------------------
# 3. Q–Q Plot
# -------------------------------
plt.figure(figsize=(6,6))
plt.plot(y_pred_mean, y_empirical, 'o', label="Predicted vs. empirical quantiles")
plt.plot([y_empirical.min(), y_empirical.max()],
         [y_empirical.min(), y_empirical.max()],
         'r--', label="1:1 line")
plt.ylabel("Empirical quantiles of true values")
plt.xlabel("Predicted quantiles (mean across samples)")
plt.title("Quantile–Quantile (Q–Q) Plot from Quantile Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(output_dir, "qq_plot_quantile_regression.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")





