import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import json
#from joblib import Parallel, delayed
#import multiprocessing
import os
import sys
import csv 
import json
from datetime import datetime
import argparse
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/')
import utils as ut

import os
import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class SmoothMultiQuantileLoss(nn.Module):
    def __init__(self, quantiles, delta: float = 1e-3):
        """
        quantiles: iterable of tau in (0,1), e.g. [0.1, 0.5, 0.9]
        delta: smoothness parameter for |u| ~= sqrt(u^2 + delta^2)
        """
        super().__init__()
        taus = torch.as_tensor(quantiles, dtype=torch.float32)
        assert torch.all((taus > 0) & (taus < 1)), "Quantiles must be in (0,1)"
        self.register_buffer("taus", taus)  # (n_quantiles,)
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch_size, n_cells, n_quantiles)
        y_true: (batch_size, n_cells)
        """
        if y_true.ndim != 2:
            raise ValueError(f"y_true must have shape (batch_size, n_cells), got {y_true.shape}")
        if y_pred.ndim != 3:
            raise ValueError(f"y_pred must have shape (batch_size, n_cells, n_quantiles), got {y_pred.shape}")

        # Expand target to match quantile dimension
        # y_true: (B, C) -> (B, C, 1)
        y_true = y_true.unsqueeze(-1)

        # Residuals per cell and quantile: (B, C, Q)
        u = y_true - y_pred

        # Smooth absolute value
        smooth_abs = torch.sqrt(u ** 2 + self.delta ** 2)

        # Broadcast taus: (1, 1, Q)
        taus = self.taus.view(1, 1, -1)

        # Smoothed pinball loss
        loss = 0.5 * (smooth_abs + (2.0 * taus - 1.0) * u)

        # Mean over batch, cells, quantiles
        return loss.mean()


class LinearMultiQuantileRegressor(nn.Module):
    def __init__(self, n_features: int, n_cells: int, n_quantiles: int):
        super().__init__()
        self.n_cells = n_cells
        self.n_quantiles = n_quantiles
        self.linear = nn.Linear(n_features, n_cells * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_features)
        returns: (batch_size, n_cells, n_quantiles)
        """
        out = self.linear(x)  # (B, C * Q)
        out = out.view(x.size(0), self.n_cells, self.n_quantiles)
        return out


def train_multi_quantile_regression_and_save(
    X: torch.Tensor,
    y: torch.Tensor,
    quantiles=(0.1, 0.5, 0.9),
    delta: float = 1e-3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    n_epochs: int = 10,
    device: str = None,
    lambda_monotonic: float = 0.0,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    csv_path: str = "training_log.csv",
    checkpoint_dir: str = "checkpoints",
    save_every: int = 5,
    random_seed: int = None,
):
    """
    Train a multi-cell, multi-quantile regression model, log losses to CSV,
    save checkpoints every N epochs, and write a global JSON with metadata.

    Expected shapes:
        X:     (n_samples, n_features)
        y:     (n_samples, n_cells)
        X_val: (n_val_samples, n_features), optional
        y_val: (n_val_samples, n_cells), optional
    """

    # --- optional seed for reproducibility ---
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Basic shape checks
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_features), got {X.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must have shape (n_samples, n_cells), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}")

    # Move data
    X = X.to(device)
    y = y.to(device)

    n_samples, n_features = X.shape
    _, n_cells = y.shape
    n_quantiles = len(quantiles)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation
    if X_val is not None and y_val is not None:
        if X_val.ndim != 2:
            raise ValueError(f"X_val must have shape (n_val_samples, n_features), got {X_val.shape}")
        if y_val.ndim != 2:
            raise ValueError(f"y_val must have shape (n_val_samples, n_cells), got {y_val.shape}")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError(
                f"X_val and y_val must have same number of samples, got {X_val.shape[0]} and {y_val.shape[0]}"
            )
        if X_val.shape[1] != n_features:
            raise ValueError(
                f"X_val must have same number of features as X, got {X_val.shape[1]} vs {n_features}"
            )
        if y_val.shape[1] != n_cells:
            raise ValueError(
                f"y_val must have same number of cells as y, got {y_val.shape[1]} vs {n_cells}"
            )

        X_val = X_val.to(device)
        y_val = y_val.to(device)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        n_val_samples = len(val_dataset)
    else:
        val_loader = None
        n_val_samples = None

    # Model & loss
    model = LinearMultiQuantileRegressor(
        n_features=n_features,
        n_cells=n_cells,
        n_quantiles=n_quantiles,
    ).to(device)

    criterion = SmoothMultiQuantileLoss(quantiles, delta=delta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_class": "LinearMultiQuantileRegressor",
        "loss_class": "SmoothMultiQuantileLoss",
        "quantiles": list(quantiles),
        "delta": float(delta),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "lambda_monotonic": float(lambda_monotonic),
        "n_epochs": int(n_epochs),
        "save_every": int(save_every),
        "n_features": int(n_features),
        "n_cells": int(n_cells),
        "n_quantiles": int(n_quantiles),
        "train_samples": int(n_samples),
        "val_samples": (int(n_val_samples) if n_val_samples is not None else None),
        "csv_path": csv_path,
        "checkpoint_dir": checkpoint_dir,
        "random_seed": int(random_seed) if random_seed is not None else None,
        "device": str(device),
        "final_epoch": None,
        "final_train_loss": None,
        "final_val_loss": None,
        "last_checkpoint": None,
    }

    last_ckpt_path = None
    last_train_loss = None
    last_val_loss = None

    for epoch in range(n_epochs):
        # --- TRAIN ---
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            preds = model(xb)         # (B, C, Q)
            loss = criterion(preds, yb)

            if lambda_monotonic > 0.0:
                # Enforce q_k <= q_{k+1} for each cell
                # preds[:, :, :-1] - preds[:, :, 1:] > 0 is a violation
                diff = preds[:, :, :-1] - preds[:, :, 1:]   # (B, C, Q-1)
                penalty = torch.relu(diff).mean()
                loss = loss + lambda_monotonic * penalty

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / n_samples

        # --- VALIDATION ---
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)   # (B, C, Q)
                    vloss = criterion(preds, yb)

                    if lambda_monotonic > 0.0:
                        diff = preds[:, :, :-1] - preds[:, :, 1:]
                        penalty = torch.relu(diff).mean()
                        vloss = vloss + lambda_monotonic * penalty

                    running_val_loss += vloss.item() * xb.size(0)

            val_loss = running_val_loss / n_val_samples
            print(
                f"Epoch {epoch + 1}/{n_epochs} "
                f"- train: {train_loss:.6f} - val: {val_loss:.6f}"
            )
        else:
            val_loss = None
            print(f"Epoch {epoch + 1}/{n_epochs} - train: {train_loss:.6f}")

        # log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

        # save checkpoint every save_every epochs
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            full_model_path = os.path.join(checkpoint_dir, f"full_model_epoch_{epoch + 1}.pt")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "quantiles": list(quantiles),
                    "delta": delta,
                    "n_features": n_features,
                    "n_cells": n_cells,
                    "n_quantiles": n_quantiles,
                    "batch_size": batch_size,
                    "lr": lr,
                    "lambda_monotonic": lambda_monotonic,
                    "random_seed": random_seed,
                },
                ckpt_path,
            )
            torch.save(model, full_model_path)

            last_ckpt_path = ckpt_path
            print(f"Saved checkpoint (and full model) at epoch {epoch + 1}.")

        last_train_loss = train_loss
        last_val_loss = val_loss

    metadata["final_epoch"] = n_epochs
    metadata["final_train_loss"] = float(last_train_loss) if last_train_loss is not None else None
    metadata["final_val_loss"] = float(last_val_loss) if last_val_loss is not None else None
    metadata["last_checkpoint"] = last_ckpt_path

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "w") as jf:
        json.dump(metadata, jf, indent=4)

    print(f"Global metadata written to: {metadata_path}")

    return model


###############
### Example ###
###############
def main():

    #######################
    ### Parse arguments ###
    #######################
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file_path", type=str)
    parser.add_argument("--q_start", type=float, default=0.05)
    parser.add_argument("--q_end", type=float, default=0.95)
    parser.add_argument("--q_n", type=int, default=19)
    parser.add_argument("--delta", type=float, default=1e-4, help="Delta for pinball loss smoothness")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--lambda_monotonic", type=float, default=0.0)
    parser.add_argument("--standardize_predictors", type=int, default=0, help="Whether to still standardize predictors or not.")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--domain", type=str, help="Domain to train model in.")
    #parser.add_argument("--", type=, default = )
    #parser.add_argument("--", type=, default = )
    #parser.add_argument("--", type=, default = )
    
    args = parser.parse_args()

    
    ####################
    ### Load my data ###
    ####################
    # load my data
    settings_file_path = args.settings_file_path #"../joint_training/v2_dpa_train_settings.json"
    
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
    
    print("Top-level keys:", list(settings.keys()))
    
    ###
    # Load temperature data
    ds = xr.open_dataset(settings['dataset_trefht'])
    print("Dataset:", settings['dataset_trefht'])

    # set train/test split
    ds_train = ds.isel(time=slice(0, 4769 * 90)) #
    ds_test = ds.isel(time=slice(4769 * 90, 476900)) #4769 * 80
    
    # transform to torch tensors
    x_tr = ut.data_to_torch(ds_train, "TREFHT")
    x_te = ut.data_to_torch(ds_test, "TREFHT")
    
    # load Z500
    ###
    # Load LE Z500 
    z500_le = xr.open_dataset(settings["dataset_z500"])
    predictors_combined_le = z500_le.pseudo_pcs.values

    #z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
    #predictors_combined_eth = z500_eth.pseudo_pcs.values 

    ## train
    train_predictors, mean_train, std_train = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
    X_torch = torch.from_numpy(train_predictors)
    

    ## validation
    validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :], mean_train, std_train)
    X_val_torch = torch.from_numpy(validation_predictors)
    print("validation set shape:", predictors_combined_le[90*4769:, :].shape, mean_train.shape, std_train.shape)


    # remove NaNs from Temperature data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)
    ###

    
    # training data
    #trefht_le_ger_mean = trefht_le.TREFHT
    #trefht_le_ger_mean
    
    # test_data
    #trefht_eth_ger_mean = trefht_eth.TREFHT
    #trefht_eth_ger_mean

    print("Predictors train shape:", X_torch.shape)
    print("Predictands train shape:", x_tr_reduced.shape)
    print("#######")
    print("Predictors test shape:", X_val_torch.shape)
    print("Predictands test shape:", x_te_reduced.shape)
    

    

    quantiles = np.linspace(args.q_start, args.q_end, args.q_n)
    

    model = train_multi_quantile_regression_and_save(
        X_torch, x_tr_reduced,
        quantiles=quantiles,
        delta=args.delta,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        n_epochs=args.n_epochs,
        lambda_monotonic=args.lambda_monotonic, # small penalty to encourage ordered quantiles 0.1 should suffice
        X_val=X_val_torch,
        y_val=x_te_reduced,
        csv_path= f"{args.save_path}training_log.csv",
        checkpoint_dir= f"{args.save_path}",
        )

    
    # Get coefficients after training
    with torch.no_grad():
        W = model.linear.weight.cpu().numpy()  # shape (n_quantiles, n_features)
        b = model.linear.bias.cpu().numpy()    # shape (n_quantiles,)
    
    print("Weights shape:", W.shape)  # (3, 1001) if 3 quantiles
    print("Bias shape:", b.shape)

    ###
    # save this script
    current_script = os.path.realpath(__file__)

    # timestamp string: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # output filename with timestamp
    out_file = os.path.join(
        args.save_path,
        f"used_baseline_training_script_{timestamp}.py"
    )

    # copy the script
    shutil.copy(current_script, out_file)

    print(f"Script saved as: {out_file}")
    ###

if __name__ == "__main__":
    main()

    

