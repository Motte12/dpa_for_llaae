import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import xarray as xr
import json
from joblib import Parallel, delayed
import multiprocessing

n_cores = multiprocessing.cpu_count()  # will show number of CPUs available inside job
print(f"Detected {n_cores} cores available to job.")

# load my data
settings_file_path = "../joint_training/v2_dpa_train_settings.json"

with open(settings_file_path, 'r') as file:
        settings = json.load(file)

# Load temperature data
trefht_le = xr.open_dataset(settings['dataset_trefht']).TREFHT
print(trefht_le)

# Load z500 data
z500_trefht = xr.open_dataset(settings['dataset_z500']).pseudo_pcs
print(z500_trefht)

trefht_flat = trefht_le.stack(feature=("lat", "lon"))
trefht_flat_clean = trefht_flat.dropna(dim="feature", how="all")
print(trefht_flat_clean.shape)

no_samples = 300
no_predictors = 20
no_predictands = 32


# Artificial data (replace with your real arrays)
X = z500_trefht[:no_samples, :no_predictors] #np.random.randn(n_samples, n_features)
Y = trefht_flat_clean[:no_samples, :no_predictands] #np.random.randn(n_samples, n_predictands)

print("Predictor shape:", X.shape)
print("Predictands shape:", Y.shape)

# -------------------------------
# 1. Fit one Quantile Ridge model per predictand
# -------------------------------
quantile = 0.5
alpha = 0.01

def fit_one_target(i):
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
    model.fit(X, Y[:, i])
    return model.intercept_, model.coef_

results = Parallel(n_jobs=n_cores, backend='threading', verbose=5)( #droping backend='threading' seemed to be faster
    delayed(fit_one_target)(i) for i in range(no_predictands)
)

intercepts, coefs = zip(*results)
intercepts = np.array(intercepts)
coefs = np.vstack(coefs)  # shape (n_predictands, n_features)

print("Fitted coefficients:", coefs.shape)

# -------------------------------
# 2. Predict all predictands
# -------------------------------
#Y_pred = X @ coefs.T + intercepts  # broadcasted addition
Y_pred = np.asarray(X) @ np.asarray(coefs).T + np.asarray(intercepts)

print("Prediction shape:", Y_pred.shape)  # no_samples, n_predictands)