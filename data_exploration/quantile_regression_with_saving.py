import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import xarray as xr
import json
from joblib import Parallel, delayed
import multiprocessing
import os

# -------------------------------
# 0. Setup
# -------------------------------
n_cores = multiprocessing.cpu_count()
print(f"Detected {n_cores} cores available to job.")

# Load paths
settings_file_path = "../joint_training/v2_dpa_train_settings.json"
with open(settings_file_path, 'r') as file:
    settings = json.load(file)

# -------------------------------
# 1. Load data
# -------------------------------
# Load temperature (predictands)
trefht_le = xr.open_dataset(settings['dataset_trefht']).TREFHT
print("TREFHT data:", trefht_le)

# Load z500 (predictors)
z500_trefht = xr.open_dataset(settings['dataset_z500']).pseudo_pcs
print("Z500 data:", z500_trefht)

# Flatten TREFHT and clean NaNs
trefht_flat = trefht_le.stack(feature=("lat", "lon"))
trefht_flat_clean = trefht_flat.dropna(dim="feature", how="all")
print("Flattened and cleaned TREFHT shape:", trefht_flat_clean.shape)

# -------------------------------
# 2. Subset data for experiment
# -------------------------------
no_samples = 4769
no_predictors = 1000
no_predictands = 648

X = z500_trefht[:no_samples, :no_predictors]
Y = trefht_flat_clean[:no_samples, :no_predictands]

print("Predictor shape:", X.shape)
print("Predictand shape:", Y.shape)

# Convert to numpy arrays
X_np = np.asarray(X)
Y_np = np.asarray(Y)

# -------------------------------
# 3. Fit one Quantile Ridge model per predictand
# -------------------------------
quantile = 0.5
alpha = 0.01

def fit_one_target(i):
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    model.fit(X_np, Y_np[:, i])
    return model.intercept_, model.coef_

results = Parallel(n_jobs=n_cores, backend='threading', verbose=5)(
    delayed(fit_one_target)(i) for i in range(no_predictands)
)

intercepts, coefs = zip(*results)
intercepts = np.array(intercepts)
coefs = np.vstack(coefs)  # shape: (n_predictands, n_features)

print("Fitted coefficients shape:", coefs.shape)

# -------------------------------
# 4. Predict all predictands
# -------------------------------
Y_pred = X_np @ coefs.T + intercepts
print("Prediction shape:", Y_pred.shape)

# -------------------------------
# 5. Save results to xarray Dataset
# -------------------------------
samples = np.arange(no_samples)
predictors = np.arange(no_predictors)
predictands = np.arange(no_predictands)

ds_results = xr.Dataset(
    {
        "coefficients": (["predictand", "predictor"], coefs),
        "intercepts":   (["predictand"], intercepts),
        "predictions":  (["sample", "predictand"], Y_pred),
    },
    coords={
        "sample": samples,
        "predictand": predictands,
        "predictor": predictors,
    },
    attrs={
        "model": "QuantileRegressor",
        "quantile": quantile,
        "alpha": alpha,
        "n_samples": no_samples,
        "n_predictors": no_predictors,
        "n_predictands": no_predictands,
        "description": "Quantile ridge regression results for TREFHT predicted from Z500 pseudo-PCs"
    }
)

# Create output folder if needed
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "quantile_regression_results.nc")

ds_results.to_netcdf(output_path)
print(f"✅ Results saved to {output_path}")
