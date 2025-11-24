import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import xarray as xr
import json
from joblib import Parallel, delayed
import torch
import pytorch_quantile_regression as pqr
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to quantile regression model to use.")
    parser.add_argument("--results_save_path", type=str, help="Path to save evaluation results.")
    parser.add_argument("--compare_model", type=str, help="Other quantile/ensemble model to compare to.")

    args = parser.parse_args()


    #############################
    ### Load comparison model ###
    #############################
    

    
    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15
    
    # cut train data

    # DPA
    #dpa_ds = xr.open_dataset(args.compare_model)#xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_100epochs/eth_ensemble_after_100_epochs/ETH_gen_dpa_ens_100_dataset_restored.nc")
    #trefht_dpa_trans_ger = dpa_ds.TREFHT.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))

    # analogues
    dpa_ds = xr.open_dataset("/work/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/analogues/analogue_ensemble_10pcs_5analogues_100analoguemembers_complete.nc").ensemble_temp.transpose("analogue_ensemble_member","time","lat","lon")
    trefht_dpa_trans_ger = dpa_ds.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    
    # calculate weighted means
    #weights
    weights_ger_pre = np.cos(np.deg2rad(trefht_dpa_trans_ger["lat"]))
    weights_ger = weights_ger_pre / weights_ger_pre.sum()
    
    # training data
    trefht_dpa_trans_ger_mean = trefht_dpa_trans_ger.weighted(weights_ger).mean(dim=("lat", "lon"))
    #trefht_dpa_trans_ger_mean

    trefht_dpa_trans_ger_mean.values.T.shape
    dpa_trans_predicted_quantiles = np.quantile(trefht_dpa_trans_ger_mean.values.T, np.linspace(0.05, 0.95, 19), axis=1).T
    print(dpa_trans_predicted_quantiles.shape)


    
    ###########################
    ### Load model and logs ###
    ###########################
    
    # Model 2
    model_path = args.model_path
    
    # Look at loss curves
    csv_path = f"{model_path}training_log.csv"
    
    # --- 1. Load parameters from metadata.json ---
    
    metadata_path = f"{model_path}metadata.json"  # adjust if needed
    print(metadata_path)
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    # load
    df = pd.read_csv(csv_path)
    
    # quick check
    print(df.head())
    # expected columns: epoch, train_loss, val_loss

    # plot loss curves
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim(0.2, 0.4)
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.results_save_path}loss_curves.png")

        
    
    # ---------------------------------------------------------
    # 1. Load quantiles & default delta from metadata.json
    # ---------------------------------------------------------
    quantiles = np.array(meta["quantiles"])
    delta_default = float(meta["delta"])
    
    print("Loaded quantiles:", quantiles)
    print("Default delta from metadata:", delta_default)
    
    # ---------------------------------------------------------
    # 2. Choose multiple delta values to plot
    # ---------------------------------------------------------
    
    # You can override or extend with your own list:
    deltas = [delta_default, 1e-1, 1e-2, 0.5]
    
    print("Plotting deltas:", deltas)
    
    # ---------------------------------------------------------
    # 3. Define smoothed pinball loss
    # ---------------------------------------------------------
    
    
    
    # ---------------------------------------------------------
    # 4. Select quantiles to visualize
    # ---------------------------------------------------------
    
    # Pick 3 representative quantiles (low, mid, high)
    if len(quantiles) > 5:
        taus_to_plot = [
            quantiles[0],
            quantiles[len(quantiles)//2],
            quantiles[-1]
        ]
    else:
        taus_to_plot = quantiles
    
    print("Quantiles used for plotting:", taus_to_plot)
    
    # ---------------------------------------------------------
    # 5. Create grid for residuals u = y - y_pred
    # ---------------------------------------------------------
    
    u = np.linspace(-3, 3, 500)   # adjust range if needed
    
    # ---------------------------------------------------------
    # 7. Optional: Multi-panel plot for different quantiles
    # ---------------------------------------------------------
    
    fig, axes = plt.subplots(1, len(taus_to_plot), figsize=(5*len(taus_to_plot), 5))
    
    if len(taus_to_plot) == 1:
        axes = [axes]
    
    for ax, tau in zip(axes, taus_to_plot):
        for delta in deltas:
            loss_vals = smooth_pinball_loss(u, tau, delta)
            ax.plot(u, loss_vals, label=f"δ = {delta}")
    
        ax.axvline(0, color="k", linestyle=":", linewidth=1)
        ax.set_title(f"τ = {tau}")
        ax.set_xlabel("Residual u")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.results_save_path}smoothed_pinball_loss.png")
    #plt.show()

    

    # ---- Load metadata ----
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
    settings_file_path = "../joint_training/v2_dpa_train_settings.json"
    
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
    
    # Load Z500 data
    z500_test = xr.open_dataset(settings['dataset_z500_eth_test']).pseudo_pcs
    z500_test_np = z500_test.values
    X_test_torch = torch.from_numpy(z500_test_np.astype("float32")).to(device)

    # ---- Predict from Qu. regression model ----
    with torch.no_grad():
        preds = model(X_test_torch)   # shape (N_test, n_quantiles)
    print(preds.shape)
    quantile_predictions = preds.cpu().numpy()

    # Temperature Test data
    trefht_eth = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    print(trefht_eth)
    
    # germany domain 
    ### Germany ###
        
    # coordinates 
    ger_lat_min = 48
    ger_lat_max = 54
    ger_lon_min = 6
    ger_lon_max = 15
    
    # cut test data
    trefht_eth_ger = trefht_eth.sel(lat=slice(ger_lat_min, ger_lat_max), lon=slice(ger_lon_min, ger_lon_max))
    print(trefht_eth_ger)
    
    # calculate weighted means
    #weights
    weights_ger_pre = np.cos(np.deg2rad(trefht_eth["lat"]))
    weights_ger = weights_ger_pre / weights_ger_pre.sum()
    
    # test_data
    trefht_eth_ger_mean = trefht_eth_ger.TREFHT.weighted(weights_ger).mean(dim=("lat", "lon"))
    #trefht_eth_ger_mean

    # prepare data for validation 
    X_test_np = z500_test_np
    y_test_np = trefht_eth_ger_mean
    quantiles = meta["quantiles"]
    # quantile regression predicted quantiles
    quantile_predictions = quantile_predictions
    
    
    # DPA predicted quantiles (ADD later)
    if args.compare_model is not None:    
        quantile_predictions_dpa = dpa_trans_predicted_quantiles

    y = y_test_np
    q_hat = quantile_predictions  # (N_test, n_quantiles)
    if args.compare_model is not None:
        q_hat_dpa = quantile_predictions_dpa
    
    coverages = []
    coverages_dpa = []
    for j, tau in enumerate(quantiles):
        # quantile regression
        coverage = np.mean(y <= q_hat[:, j])
        coverages.append(coverage)
        print(f"Qu.regression: tau={tau:.2f}, empirical coverage={coverage:.4f}")
    
        # comparison model (DPA, analogues)
        if args.compare_model is not None:
            coverage_dpa = np.mean(y <= q_hat_dpa[:, j])
            coverages_dpa.append(coverage_dpa)
            print(f"Qu.regression: tau={tau:.2f}, empirical coverage DPA={coverage_dpa:.4f}")
    
    plt.figure()
    plt.plot(quantiles, coverages, marker="o", label="empirical coverage qu. regression")
    if args.compare_model is not None:
        plt.plot(quantiles, coverages_dpa, marker="o", label="empirical coverage comparison model (eg DPA or analogues)")
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlabel("Nominal quantile (tau)")
    plt.ylabel("Empirical coverage P(y <= q̂_tau)")
    plt.title("Quantile calibration")
    plt.legend()
    plt.savefig(f"{args.results_save_path}empirical_coverage_curves.png")
    #plt.show()

    #####################
    ### PIT histogram ###
    #####################
    
    # quantile regression
    u = compute_pit_from_quantiles(y_test_np, quantile_predictions, quantiles)
    ax = plot_pit_histogram(u, bins=20, title = "PIT Histogram quantile regression")
    plt.ylim(0,5)
    plt.savefig(f"{args.results_save_path}PIT_hist_qu_regression.png")
    #plt.show()

    if args.compare_model is not None:
        # DPA
        u = compute_pit_from_quantiles(y_test_np, quantile_predictions_dpa, quantiles)
        ax = plot_pit_histogram(u, bins=20, title = "PIT Histogram comparison model")
        plt.ylim(0,5)
        plt.savefig(f"{args.results_save_path}PIT_hist_comparison_model.png")
        #plt.show()


    #############################
    ### Pinball/Quantile loss ###
    #############################
    loss = pinball_loss_multi_np(y_test_np, quantile_predictions, quantiles=quantiles)
    print("Overall mean pinball loss:", loss)

    per_q = pinball_loss_per_quantile_np(y_test_np, quantile_predictions, quantiles=quantiles)
    #plt.scatter(quantiles, per_q, marker = 'x', s=10, color = "tab:blue", label = "Quantile regression")

    ### DPA
    if args.compare_model is not None:
        print("quantile preds DPA shape:", quantile_predictions_dpa.shape)
        loss_dpa = pinball_loss_multi_np(y_test_np, quantile_predictions_dpa, quantiles=quantiles)
        print("loss dpa shape:", loss_dpa.shape)
        per_q_dpa = pinball_loss_per_quantile_np(y_test_np, quantile_predictions_dpa, quantiles=quantiles)



    ############
    ### CRPS ###
    ############
    ### Quantile regression
    qr_crps_mean, qr_crps_per_sample = crps_from_quantiles(
        y_test_np, quantile_predictions, quantiles
    )
    print("QR model mean CRPS:", qr_crps_mean)

    ###########
    ### DPA ###
    ###########
    qr_crps_mean_dpa, qr_crps_per_sample_dpa = crps_from_quantiles(
        y_test_np, quantile_predictions_dpa, quantiles
    )

    #################
    ### Sharpness ###
    #################
    ### Quantile Regression
    qr_spreads, qr_spread_summary = sharpness_qr(
        quantile_predictions, quantiles, lower=0.1, upper=0.9
    )
    print("QR sharpness summary:", qr_spread_summary)

    ### Comparison model (DPA)
    ###########
    ### DPA ###
    ###########
    qr_spreads_dpa, qr_spread_summary_dpa = sharpness_qr(
        quantile_predictions_dpa, quantiles, lower=0.1, upper=0.9
    )

    ########################
    ### put all together ###
    ########################

    ###################################
    ### Summary quantile regression ###
    ###################################
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # upper Left: calibration curve
    plot_calibration_curve(y_test_np, quantile_predictions, quantiles, ax=axes[0,0])
    
    # upper Right: PIT histogram
    u = compute_pit_from_quantiles(y_test_np, quantile_predictions, quantiles)
    plot_pit_histogram(u, bins=20, ax=axes[0,1])
    
    # lower left
    _, _, spreads, mean_spread, median_spread, q10, q90 = plot_qr_spreads(qr_spreads, axes[1,0], bins=40)
    
    # lower right
    axes[1,1].scatter(quantiles, per_q, marker = 'x', s=20, color = "tab:purple", label = "DPA")
    
    
    #plt.tight_layout()
    ###
    # assuming you already computed these:
    # spreads, mean_spread, median_spread, q10, q90
    
    summary = (
        f"Quantile regression\n"
        f"Overall mean pinball loss: {loss}\n"
        "\n"
        f"CRPS: {qr_crps_mean}\n"
        "\n"
        "QR spread summary:\n"
        f"  N samples      : {spreads.size}\n"
        f"  mean spread    : {mean_spread:.6f}\n"
        f"  median spread  : {median_spread:.6f}\n"
        f"  10% spread     : {q10:.6f}\n"
        f"  90% spread     : {q90:.6f}"
    )
    
    
    
    # Clear anything already in that axis (optional)
    axes[2,0].cla()
    
    axes[2,0].text(
        0, 0.8,              # position in axis coordinates (left, top)
        summary,
        va='top', ha='left',
        fontsize=14,
        family='monospace',    # keeps columns aligned
        transform=axes[2,0].transAxes
    )
    
    axes[2,0].set_axis_off()  # optional: hides the plot frame
    axes[2,1].set_axis_off()
    plt.savefig(f"{args.results_save_path}summary_qu_regression.png")
    #plt.show()

    
    ###################
    ### Summary DPA ###
    ###################
    if args.compare_model is not None:
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
        # upper Left: calibration curve
        plot_calibration_curve(y_test_np, quantile_predictions_dpa, quantiles, ax=axes[0,0])
        
        # upper Right: PIT histogram
        u = compute_pit_from_quantiles(y_test_np, quantile_predictions_dpa, quantiles)
        plot_pit_histogram(u, bins=20, ax=axes[0,1])
        
        # lower left
        _, _, spreads_dpa, mean_spread_dpa, median_spread_dpa, q10_dpa, q90_dpa = plot_qr_spreads(qr_spreads_dpa, axes[1,0], bins=40)
        
        # lower right
        print("loss dpa:", loss_dpa.shape)
        print("quantiles shape:", len(quantiles))
        axes[1,1].scatter(quantiles, per_q_dpa, marker = 'x', s=20, color = "tab:purple", label = "DPA")
        
        
        #plt.tight_layout()
        ###
        # assuming you already computed these:
        # spreads, mean_spread, median_spread, q10, q90
        
        summary = (
            f"Comparison Model (DPA)\n"
            f"Overall mean pinball loss: {loss_dpa}\n"
            "\n"
            f"CRPS: {qr_crps_mean_dpa}\n"
            "\n"
            "QR spread summary:\n"
            f"  N samples      : {spreads_dpa.size}\n"
            f"  mean spread    : {mean_spread_dpa:.6f}\n"
            f"  median spread  : {median_spread_dpa:.6f}\n"
            f"  10% spread     : {q10_dpa:.6f}\n"
            f"  90% spread     : {q90_dpa:.6f}"
        )
        
        
        
        # Clear anything already in that axis (optional)
        axes[2,0].cla()
        
        axes[2,0].text(
            0, 0.8,              # position in axis coordinates (left, top)
            summary,
            va='top', ha='left',
            fontsize=14,
            family='monospace',    # keeps columns aligned
            transform=axes[2,0].transAxes
        )
        
        axes[2,0].set_axis_off()  # optional: hides the plot frame
        axes[2,1].set_axis_off()
        plt.savefig(f"{args.results_save_path}summary_dpa.png")

    ####################
    ### Summary Both ###
    ####################

    if args.compare_model is not None:
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
        # 1) CALIBRATION CURVE (top-left): QR vs DPA
        ax_cal = axes[0, 0]
    
        cover_qr  = compute_coverage_per_quantile(y_test_np, quantile_predictions,     quantiles)
        cover_dpa = compute_coverage_per_quantile(y_test_np, quantile_predictions_dpa, quantiles)
    
        ax_cal.plot(quantiles, cover_qr,  marker="o", label="QR")
        ax_cal.plot(quantiles, cover_dpa, marker="s", label="DPA")
        ax_cal.plot([0, 1], [0, 1], "k--", label="Ideal 1:1")
    
        ax_cal.set_xlabel("Nominal quantile τ")
        ax_cal.set_ylabel("Empirical coverage P(y ≤ q̂τ(x))")
        ax_cal.set_title("Quantile calibration (QR vs DPA)")
        ax_cal.grid(True)
        ax_cal.legend()
    
        # 2) PIT HISTOGRAM (top-right): QR vs DPA
        ax_pit = axes[0, 1]
    
        u_qr  = compute_pit_from_quantiles(y_test_np, quantile_predictions,     quantiles)
        u_dpa = compute_pit_from_quantiles(y_test_np, quantile_predictions_dpa, quantiles)
    
        ax_pit.hist(u_qr,  bins=20, range=(0, 1), density=True, alpha=0.5, edgecolor="black",
                    label="QR")
        ax_pit.hist(u_dpa, bins=20, range=(0, 1), density=True, alpha=0.5, edgecolor="black",
                    label="DPA")
        ax_pit.hlines(1.0, 0, 1, colors="red", linestyles="--", label="Uniform(0,1)")
    
        ax_pit.set_xlabel("PIT value u")
        ax_pit.set_ylabel("Density")
        ax_pit.set_title("PIT histograms (QR vs DPA)")
        ax_pit.set_xlim(0, 1)
        ax_pit.grid(True)
        ax_pit.legend()
    
        # 3) SHARPNESS / SPREADS (middle-left): QR vs DPA
        ax_spread = axes[1, 0]
    
        # You already computed these earlier:
        # qr_spreads, qr_spread_summary = sharpness_qr(quantile_predictions,     quantiles, ...)
        # qr_spreads_dpa, qr_spread_summary_dpa = sharpness_qr(quantile_predictions_dpa, quantiles, ...)
        ax_spread.hist(qr_spreads,      bins=40, alpha=0.5, edgecolor="black", label="QR")
        ax_spread.hist(qr_spreads_dpa,  bins=40, alpha=0.5, edgecolor="black", label="DPA")
    
        mean_spread_qr  = qr_spread_summary["mean_spread"]
        mean_spread_dpa = qr_spread_summary_dpa["mean_spread"]
    
        ax_spread.axvline(mean_spread_qr,  color="C0", linestyle="--",
                          label=f"QR mean={mean_spread_qr:.3f}")
        ax_spread.axvline(mean_spread_dpa, color="C1", linestyle="--",
                          label=f"DPA mean={mean_spread_dpa:.3f}")
    
        ax_spread.set_title("Sharpness: Conditional spread distribution (QR vs DPA)")
        ax_spread.set_xlabel("Spread (q_upper - q_lower)")
        ax_spread.set_ylabel("Count")
        ax_spread.grid(True, alpha=0.3)
        ax_spread.set_xlim(0, 5)
        ax_spread.legend()
    
        # 4) PINBALL LOSS PER QUANTILE (middle-right): QR vs DPA
        ax_pinball = axes[1, 1]
    
        # per_q, per_q_dpa already computed:
        # per_q      = pinball_loss_per_quantile_np(y_test_np, quantile_predictions,     quantiles)
        # per_q_dpa  = pinball_loss_per_quantile_np(y_test_np, quantile_predictions_dpa, quantiles)
        ax_pinball.plot(quantiles, per_q,     marker="o", label="QR")
        ax_pinball.plot(quantiles, per_q_dpa, marker="s", label="DPA")
    
        ax_pinball.set_xlabel("Quantile τ")
        ax_pinball.set_ylabel("Mean pinball loss")
        ax_pinball.set_title("Per-quantile pinball loss (QR vs DPA)")
        ax_pinball.grid(True)
        ax_pinball.legend()
    
        # 5) METRICS TEXT PANELS (bottom row)
        # left: QR, right: DPA
        axes[2, 0].cla()
        axes[2, 1].cla()
    
        n_qr  = qr_spreads.size
        n_dpa = qr_spreads_dpa.size
    
        summary_qr = (
            "Quantile regression (QR)\n"
            f"Overall mean pinball loss: {loss:.6f}\n"
            f"CRPS: {qr_crps_mean:.6f}\n"
            "\n"
            "QR spread summary:\n"
            f"  N samples      : {n_qr}\n"
            f"  mean spread    : {mean_spread_qr:.6f}\n"
            f"  median spread  : {qr_spread_summary['median_spread']:.6f}\n"
            f"  10% spread     : {qr_spread_summary['lower_quantile_spread']:.6f}\n"
            f"  90% spread     : {qr_spread_summary['upper_quantile_spread']:.6f}"
        )
    
        summary_dpa = (
            "Comparison model (DPA)\n"
            f"Overall mean pinball loss: {loss_dpa:.6f}\n"
            f"CRPS: {qr_crps_mean_dpa:.6f}\n"
            "\n"
            "QR spread summary:\n"
            f"  N samples      : {n_dpa}\n"
            f"  mean spread    : {mean_spread_dpa:.6f}\n"
            f"  median spread  : {qr_spread_summary_dpa['median_spread']:.6f}\n"
            f"  10% spread     : {qr_spread_summary_dpa['lower_quantile_spread']:.6f}\n"
            f"  90% spread     : {qr_spread_summary_dpa['upper_quantile_spread']:.6f}"
        )
    
        axes[2, 0].text(
            0.0, 0.9, summary_qr,
            va="top", ha="left",
            fontsize=12,
            family="monospace",
            transform=axes[2, 0].transAxes,
        )
        axes[2, 0].set_axis_off()
    
        axes[2, 1].text(
            0.0, 0.9, summary_dpa,
            va="top", ha="left",
            fontsize=12,
            family="monospace",
            transform=axes[2, 1].transAxes,
        )
        axes[2, 1].set_axis_off()
    
        plt.tight_layout()
        plt.savefig(f"{args.results_save_path}summary_qr_vs_dpa.png")
        # plt.show()









def smooth_pinball_loss(u, tau, delta):
        """
        Smoothed pinball loss:
            rho_τ^δ(u) = 0.5 * ( sqrt(u^2 + δ^2) + (2τ - 1)*u )
        """
        u = np.asarray(u)
        smooth_abs = np.sqrt(u**2 + delta**2)
        return 0.5 * (smooth_abs + (2*tau - 1.0) * u)

def compute_pit_from_quantiles(y_true, q_preds, quantiles):
    """
    Compute PIT values using piecewise linear interpolation in (τ, q̂τ(x)) space.

    y_true: (N,)
    q_preds: (N, Q) predicted quantiles for each sample
    quantiles: (Q,)
    Returns: u: (N,) PIT values in [0,1]
    """
    y_true = np.asarray(y_true).reshape(-1)
    q_preds = np.asarray(q_preds)
    taus = np.asarray(quantiles)

    N, Q = q_preds.shape
    assert Q == len(taus)

    u = np.zeros(N, dtype=float)

    for i in range(N):
        y = y_true[i]
        qs = q_preds[i, :]

        # If y below smallest predicted quantile
        if y <= qs[0]:
            u[i] = taus[0]
            continue

        # If y above largest predicted quantile
        if y >= qs[-1]:
            u[i] = taus[-1]
            continue

        # Find interval qs[k-1] <= y <= qs[k]
        idx = np.searchsorted(qs, y)
        q_low, q_high = qs[idx - 1], qs[idx]
        tau_low, tau_high = taus[idx - 1], taus[idx]

        # Linear interpolation in quantile space
        if q_high == q_low:
            # Degenerate case: quantiles equal; fall back to mid-τ
            u[i] = 0.5 * (tau_low + tau_high)
        else:
            frac = (y - q_low) / (q_high - q_low)
            u[i] = tau_low + frac * (tau_high - tau_low)

    return u

def plot_pit_histogram(u, bins=20, ax=None, title=""):
    """
    Plot PIT histogram.

    u: (N,) PIT values in [0,1]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(u, bins=bins, range=(0, 1), density=True, alpha=0.7, edgecolor="black")
    # Reference uniform density line
    ax.hlines(1.0, 0, 1, colors="red", linestyles="--", label="Uniform(0,1)")

    ax.set_xlabel("PIT value u")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True)

    return ax

    # Pinball/Quantile Loss

def pinball_loss_multi_np(y_true, y_pred, quantiles):
    """
    Pinball loss for multiple quantiles.

    y_true: array (N,)
    y_pred: array (N, Q)
    quantiles: list/array of Q quantile levels
    """
    y_true = np.asarray(y_true).reshape(-1, 1)    # (N,1)
    y_pred = np.asarray(y_pred)                  # (N,Q)
    taus = np.asarray(quantiles).reshape(1, -1)  # (1,Q)

    e = y_true - y_pred                          # (N,Q)

    loss = np.maximum(taus * e, (taus - 1.0) * e)  # (N,Q)

    return loss.mean()   # scalar


def pinball_loss_per_quantile_np(y_true, y_pred, quantiles):
    """
    Returns per-quantile pinball loss.

    y_true: array (N,)
    y_pred: array (N, Q)
    quantiles: list of Q quantile levels
    """
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    taus = np.asarray(quantiles).reshape(1, -1)

    e = y_true - y_pred
    loss = np.maximum(taus * e, (taus - 1) * e)  # (N,Q)

    return loss.mean(axis=0)  # shape (Q,)


def pinball_loss_matrix(y_true, q_preds, taus):
    """
    Compute pinball loss ρ_τ(y - q_τ(x)) for all samples and quantiles.

    y_true: (N,)
    q_preds: (N, Q)
    taus: (Q,)
    Returns: loss matrix of shape (N, Q)
    """
    y_true = np.asarray(y_true).reshape(-1, 1)   # (N,1)
    q_preds = np.asarray(q_preds)               # (N,Q)
    taus = np.asarray(taus).reshape(1, -1)      # (1,Q)

    e = y_true - q_preds                        # (N,Q)
    loss = np.maximum(taus * e, (taus - 1.0) * e)
    return loss

def crps_from_quantiles(y_true, q_preds, taus):
    """
    Approximate CRPS from quantile regression outputs via:
        CRPS ≈ ∫_0^1 ρ_τ(y - q_τ) dτ  (discretized in τ)

    y_true: (N,)
    q_preds: (N, Q)
    taus: (Q,)
    Returns:
        crps_mean: scalar (mean CRPS over all samples)
        crps_per_sample: (N,) CRPS per sample
    """
    taus = np.asarray(taus)
    # ensure sorted taus
    sort_idx = np.argsort(taus)
    taus_sorted = taus[sort_idx]
    q_sorted = q_preds[:, sort_idx]

    # pinball loss matrix: (N,Q)
    loss_mat = pinball_loss_matrix(y_true, q_sorted, taus_sorted)  # (N,Q)

    # trapezoidal integration over τ axis
    # for each sample, integrate loss(τ) dτ
    # shape of loss_mat: (N,Q)
    crps_per_sample = np.trapezoid(loss_mat, x=taus_sorted, axis=1)  # (N,)

    crps_mean = crps_per_sample.mean()
    return crps_mean, crps_per_sample

# CRPS for ensemble model (exact for empirical distribution)
def crps_ensemble(y_true, ens_samples):
    """
    CRPS for an ensemble predictive distribution:

        CRPS(F, y) = E|X - y| - 0.5 E|X - X'|

    y_true: (N,)
    ens_samples: (N, K) predictive samples for each x

    Returns:
        crps_mean: scalar
        crps_per_sample: (N,)
    """
    y_true = np.asarray(y_true).reshape(-1)
    ens_samples = np.asarray(ens_samples)
    N, K = ens_samples.shape

    crps_vals = np.zeros(N, dtype=float)

    for i in range(N):
        y = y_true[i]
        s = ens_samples[i, :]              # (K,)

        # term1 = E|X - y|
        term1 = np.mean(np.abs(s - y))

        # term2 = 0.5 E|X - X'|
        # pairwise absolute differences, shape (K,K)
        diff = np.abs(s[:, None] - s[None, :])
        term2 = 0.5 * np.mean(diff)

        crps_vals[i] = term1 - term2

    crps_mean = crps_vals.mean()
    return crps_mean, crps_vals

def sharpness_qr(q_preds, taus, lower=0.1, upper=0.9):
    """
    Compute conditional spread for QR model:
        spread_i = q_upper(x_i) - q_lower(x_i)

    q_preds: (N, Q)
    taus: (Q,)
    lower, upper: quantile levels for spread (e.g. 0.1, 0.9)

    Returns:
        spreads: (N,)
        summary: dict with mean, median, etc.
    """
    taus = np.asarray(taus)
    q_preds = np.asarray(q_preds)

    # find nearest quantile indices
    lower_idx = np.argmin(np.abs(taus - lower))
    upper_idx = np.argmin(np.abs(taus - upper))

    q_lower = q_preds[:, lower_idx]
    q_upper = q_preds[:, upper_idx]

    spreads = q_upper - q_lower

    summary = {
        "mean_spread": spreads.mean(),
        "median_spread": np.median(spreads),
        "lower_quantile_spread": np.quantile(spreads, 0.1),
        "upper_quantile_spread": np.quantile(spreads, 0.9),
    }

    return spreads, summary

def sharpness_ensemble(ens_samples, lower=0.1, upper=0.9):
    """
    Sharpness from ensemble samples:

        spread_i = q_upper(samples_i) - q_lower(samples_i)

    ens_samples: (N, K)
    """
    ens_samples = np.asarray(ens_samples)
    lower_q = np.quantile(ens_samples, lower, axis=1)  # (N,)
    upper_q = np.quantile(ens_samples, upper, axis=1)  # (N,)

    spreads = upper_q - lower_q

    summary = {
        "mean_spread": spreads.mean(),
        "median_spread": np.median(spreads),
        "lower_quantile_spread": np.quantile(spreads, 0.1),
        "upper_quantile_spread": np.quantile(spreads, 0.9),
    }

    return spreads, summary

def plot_qr_spreads(
    qr_spreads,
    ax=None,
    bins: int = 40,
    title: str = "QR conditional spread distribution",
    xlabel: str = "Spread (q_upper - q_lower)",
    figsize=(8, 5),
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot a histogram of QR spreads and print summary statistics.

    Parameters
    ----------
    qr_spreads : array-like, shape (N,)
        Per-sample conditional spread values, e.g. from sharpness_qr().
    bins : int, optional
        Number of histogram bins.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    figsize : tuple, optional
        Figure size (width, height).
    show : bool, optional
        If True, call plt.show() at the end.
    save_path : str or None, optional
        If provided, save the figure to this path (e.g. "qr_spreads.png").
    """
    spreads = np.asarray(qr_spreads).reshape(-1)

    # --- summary statistics ---
    mean_spread = spreads.mean()
    median_spread = np.median(spreads)
    q10 = np.quantile(spreads, 0.10)
    q90 = np.quantile(spreads, 0.90)

    print("QR spread summary:")
    print(f"  N samples      : {spreads.size}")
    print(f"  mean spread    : {mean_spread:.6f}")
    print(f"  median spread  : {median_spread:.6f}")
    print(f"  10% spread     : {q10:.6f}")
    print(f"  90% spread     : {q90:.6f}")

    # --- histogram plot ---
    
    fig=None
    if ax is None:
        fig, ax_pre = plt.subplots(figsize=figsize)
        ax = ax_pre

    ax.hist(spreads, bins=bins, alpha=0.7, edgecolor="black")
    ax.axvline(mean_spread, color="red", linestyle="--", linewidth=2,
               label=f"mean = {mean_spread:.3f}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0,5)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")

    #if show:
        #plt.show()

    return fig, ax, spreads, mean_spread, median_spread, q10, q90

def compute_coverage_per_quantile(y_true, q_preds, quantiles):
    """
    y_true: (N,)
    q_preds: (N, Q)
    quantiles: (Q,)
    Returns: empirical coverages array of shape (Q,)
    """
    y_true = np.asarray(y_true).reshape(-1)
    q_preds = np.asarray(q_preds)
    quantiles = np.asarray(quantiles)

    coverages = []
    for j in range(len(quantiles)):
        tau = quantiles[j]
        q_tau = q_preds[:, j]
        cov = np.mean(y_true <= q_tau)
        coverages.append(cov)

    return np.array(coverages)


def plot_calibration_curve(y_true, q_preds, quantiles, ax=None):
    """
    Plot nominal quantile vs empirical coverage.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    quantiles = np.asarray(quantiles)
    coverages = compute_coverage_per_quantile(y_true, q_preds, quantiles)

    ax.plot(quantiles, coverages, marker="o", label="Empirical coverage")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal 1:1")  # 1-1 line

    ax.set_xlabel("Nominal quantile τ")
    ax.set_ylabel("Empirical coverage P(y ≤ q̂τ(x))")
    ax.set_title("Quantile calibration curve")
    ax.grid(True)
    ax.legend()

    return ax




    
    
if __name__ == "__main__":
    main()