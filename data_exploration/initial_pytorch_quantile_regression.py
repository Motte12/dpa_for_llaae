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
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut

class SmoothMultiQuantileLoss(nn.Module):
    def __init__(self, quantiles, delta: float = 1e-3):
        """
        quantiles: iterable of τ in (0,1), e.g. [0.1, 0.5, 0.9]
        delta: smoothness parameter for |u| ≈ sqrt(u^2 + δ^2)
        """
        super().__init__()
        taus = torch.as_tensor(quantiles, dtype=torch.float32)
        assert torch.all((taus > 0) & (taus < 1)), "Quantiles must be in (0,1)"
        self.register_buffer("taus", taus)  # (n_quantiles,)
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch_size, n_quantiles)
        y_true: (batch_size,) or (batch_size, 1)
        """
        # Ensure shape (batch_size, 1)
        y_true = y_true.view(-1, 1)

        # Residuals per quantile: (batch_size, n_quantiles)
        u = y_true - y_pred

        # Smooth |u|
        smooth_abs = torch.sqrt(u ** 2 + self.delta ** 2)

        # Broadcast taus: (1, n_quantiles)
        taus = self.taus.view(1, -1)

        # ρ_τ^δ(u) = 0.5 * ( smooth_abs(u) + (2τ - 1) * u )
        loss = 0.5 * (smooth_abs + (2 * taus - 1.0) * u)

        # Mean over batch and quantiles
        return loss.mean()

class LinearMultiQuantileRegressor(nn.Module):
    def __init__(self, n_features: int, n_quantiles: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_features)
        # returns: (batch_size, n_quantiles)
        return self.linear(x)




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
    Train a multi-quantile regression model, log losses to CSV,
    save checkpoints every N epochs, and write a single global
    JSON with all important parameters for reproducibility.
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

    # Move data
    X = X.to(device)
    y = y.to(device).view(-1)
    n_samples, n_features = X.shape
    n_quantiles = len(quantiles)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation
    if X_val is not None and y_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device).view(-1)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        n_val_samples = len(val_dataset)
    else:
        val_loader = None
        n_val_samples = None

    # Model & loss
    model = LinearMultiQuantileRegressor(n_features, n_quantiles).to(device)
    criterion = SmoothMultiQuantileLoss(quantiles, delta=delta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # --- prepare global metadata (we'll fill some fields later) ---
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
        "n_quantiles": int(n_quantiles),
        "train_samples": int(n_samples),
        "val_samples": (int(n_val_samples) if n_val_samples is not None else None),
        "csv_path": csv_path,
        "checkpoint_dir": checkpoint_dir,
        "random_seed": int(random_seed) if random_seed is not None else None,
        "device": str(device),
        # to be filled at the end:
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
            preds = model(xb)
            loss = criterion(preds, yb)

            if lambda_monotonic > 0.0:
                diff = preds[:, :-1] - preds[:, 1:]
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
                    preds = model(xb)
                    vloss = criterion(preds, yb)
                    running_val_loss += vloss.item() * xb.size(0)

            val_loss = running_val_loss / n_val_samples
            print(
                f"Epoch {epoch+1}/{n_epochs} "
                f"- train: {train_loss:.6f} - val: {val_loss:.6f}"
            )
        else:
            val_loss = None
            print(f"Epoch {epoch+1}/{n_epochs} - train: {train_loss:.6f}")

        # log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

        # save checkpoint every save_every epochs
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            full_model_path = os.path.join(
                checkpoint_dir, f"full_model_epoch_{epoch+1}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "quantiles": quantiles,
                    "delta": delta,
                    "n_features": n_features,
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
            print(f"Saved checkpoint (and full model) at epoch {epoch+1}.")

        # remember last epoch metrics
        last_train_loss = train_loss
        last_val_loss = val_loss

    # --- fill final metadata and write ONE global JSON ---
    metadata["final_epoch"] = n_epochs
    metadata["final_train_loss"] = float(last_train_loss) if last_train_loss is not None else None
    metadata["final_val_loss"] = float(last_val_loss) if last_val_loss is not None else None
    metadata["last_checkpoint"] = last_ckpt_path

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(metadata_path, "w") as jf:
        json.dump(metadata, jf, indent=4)

    print(f"Global metadata written to: {metadata_path}")

    return model




def initial_train_multi_quantile_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    quantiles=(0.1, 0.5, 0.9),
    delta: float = 1e-4,
    batch_size: int = 4096,
    lr: float = 1e-3,
    n_epochs: int = 10,
    device: str = None,
    lambda_monotonic: float = 0.0,  # optional penalty (see below)
    ):
    """
    X: (n_samples, n_features) float32 tensor
    y: (n_samples,) or (n_samples, 1) float32 tensor
    quantiles: iterable of τ's, e.g. (0.1, 0.5, 0.9)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X = X.to(device)
    y = y.to(device).view(-1)

    n_samples, n_features = X.shape
    n_quantiles = len(quantiles)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LinearMultiQuantileRegressor(n_features, n_quantiles).to(device)
    criterion = SmoothMultiQuantileLoss(quantiles, delta=delta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)  # (batch, n_quantiles)

            loss = criterion(preds, yb)

            # OPTIONAL: monotonicity penalty so q_{τ1} <= q_{τ2} <= ...
            if lambda_monotonic > 0.0:
                # differences between consecutive quantiles
                diff = preds[:, :-1] - preds[:, 1:]  # (batch, n_quantiles-1)
                # only penalize violations (where lower-q > higher-q)
                penalty = torch.relu(diff).mean()
                loss = loss + lambda_monotonic * penalty

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / n_samples
        print(f"Epoch {epoch+1}/{n_epochs} - loss: {epoch_loss:.6f}")

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
    
    # Load LE temperature data
    # Train data
    trefht_le = xr.open_dataset(settings['dataset_trefht'])
    print(trefht_le)
    
    # Load LE Z500 
    z500_le = xr.open_dataset(settings["dataset_z500"])
    #if args.standardize_predictors:
    #    predictors_combined_le, _, _ = ut.standardize_numpy(z500_le.pseudo_pcs.values) 
    #else:
    #    predictors_combined_le = z500_le.pseudo_pcs.values #xr.concat([gmt_le_expanded, z500_le.pseudo_pcs], dim="mode")

    predictors_combined_le = z500_le.pseudo_pcs.values

    print(z500_le)
    
    
    # Load ETH ensemble data
    # Test data
    trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    print(trefht_eth)
    
    z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
    print(z500_eth)

    # Load GMT ETH
    #gmt_eth_pre = xr.open_dataset(settings["gmt_eth"])
    #gmt_eth = (gmt_eth_pre - gmt_eth_pre.mean()) / gmt_eth_pre.std()
    
    # concatenate data 
    #gmt_eth_expanded = gmt_eth.TREFHT.expand_dims(mode=[-1]).T  # add 'mode' dimension with length 1
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
    trefht_le_ger_mean
    
    # test_data
    trefht_eth_ger_mean = trefht_eth_ger.TREFHT.weighted(weights_ger).mean(dim=("lat", "lon"))
    trefht_eth_ger_mean

    print("Predictors train shape:", predictors_combined_le.shape)
    print("Predictands train shape:", trefht_le_ger_mean.shape)
    print("#######")
    print("Predictors train shape:", predictors_combined_eth.shape)
    print("Predictands test shape:", trefht_eth_ger_mean.shape)


    # split 

    if args.standardize_predictors:
        ###
        print("Predictors are being standardized")
        ## train
        train_predictors, train_mean, train_std = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
        X_torch = torch.from_numpy(train_predictors)
        y_torch = torch.from_numpy(trefht_le_ger_mean.values)[:90*4769]
    
        ## validation
        validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :], train_mean, train_std)
        X_val_torch = torch.from_numpy(validation_predictors)
        y_val_torch = torch.from_numpy(trefht_le_ger_mean.values)[90*4769:]
    
    else:
        ## train
        X_torch = torch.from_numpy(predictors_combined_le)[:90*4769, :]
        y_torch = torch.from_numpy(trefht_le_ger_mean.values)[:90*4769]
    
        ## validation
        X_val_torch = torch.from_numpy(predictors_combined_le)[90*4769:, :]
        y_val_torch = torch.from_numpy(trefht_le_ger_mean.values)[90*4769:]

    
    
    print("X:", X_torch.shape)
    print("y:", y_torch.shape)

    print("X_val:", X_val_torch.shape)
    print("y_val:", y_val_torch.shape)

    
    # Example fake data ##############################################
    #n_samples = 476900
    #n_features = 1001
    
    #X_np = np.random.randn(n_samples, n_features).astype("float32")
    # Example: y depends on first 3 features + noise
    #y_np = (2.0 * X_np[:, 0] - 1.5 * X_np[:, 1] + 0.5 * X_np[:, 2] 
    #        + 0.5 * np.random.randn(n_samples)).astype("float32")
    
    #X_torch = torch.from_numpy(X_np) #shape (sample, features)
    #y_torch = torch.from_numpy(y_np)
    ###################################################################

    # define quantiles
    # quantiles = np.linspace(0.05, 0.95, 19)
    quantiles = np.linspace(args.q_start, args.q_end, args.q_n)
    
    #model = train_multi_quantile_regression(
    #    X_torch,
    #    y_torch,
    #    quantiles=quantiles,
    #    delta=1e-2,          # start a bit larger; you can lower later
    #    batch_size=8192,     # increase if GPU memory allows
    #    lr=1e-3,
    #    n_epochs=10,
    #    lambda_monotonic= 0.0 #0.1 # small penalty to encourage ordered quantiles 0.1 should suffice
    #)

    model = train_multi_quantile_regression_and_save(
        X_torch, y_torch,
        quantiles=quantiles,
        delta=args.delta,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        n_epochs=args.n_epochs,
        lambda_monotonic=args.lambda_monotonic, # small penalty to encourage ordered quantiles 0.1 should suffice
        X_val=X_val_torch,
        y_val=y_val_torch,
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

    

