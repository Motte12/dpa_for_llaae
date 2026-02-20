import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut
import evaluation
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description="Example script with arguments")

    # Define arguments
    parser.add_argument("--model_path", type=str, help="Directory that contains model folder.")
    parser.add_argument("--model", type=str, help="Name of model folder.")
    parser.add_argument("--settings_file_path", type=str, help="Settings containing LE and ETH data.")
    parser.add_argument("--settings_era5_path", type=str, help="Settings containing ERA5 settings.")
    parser.add_argument("--validation_truth_save_path", type=str, help=".")
    parser.add_argument("--eth_cf_truth_save_path", type=, help=".")
    parser.add_argument("--eth_fact_truth_save_path", type=, help=".")
    parser.add_argument("--era5_fact_truth_save_path", type=, help=".")
    parser.add_argument("--era5_cf_truth_save_path", type=, help=".")
    parser.add_argument("--eval_no_epochs", type=, help=".")
    parser.add_argument("--save_path", type=, help=".")
    parser.add_argument("--", type=, help=".")
    parser.add_argument("--", type=, help=".")
    parser.add_argument("--", type=, help=".")
    parser.add_argument("--", type=, help=".")
    args = parser.parse_args()

    # setup log file
    with open(f"{model_path}/{model}/test_bias_evaluation.txt", "w") as logfile:
        sys.stdout = logfile
    
    
    #####################
    ### True Datasets ###
    #####################
    
    ############################
    ### Validation truth set ### (save these once!!!, create script for it)
    ############################
    test_val = np.load(args.validation_truth_save_path, mmap_mode="r")

    ########################
    ### ETH Test Factual ###
    ########################
        
    # assign arrays
    x_eth_fact_reduced_mean_numpy = np.load(args.eth_fact_truth_save_path, mmap_mode="r") #x_eth_fact_reduced_mean.detach().cpu().numpy()
    
    ###############################
    ### ETH Test Counterfactual ###
    ###############################
    
    # assign arrays
    x_eth_cf_reduced_mean_numpy = np.load(args.eth_cf_truth_save_path, mmap_mode="r") #x_eth_cf_reduced_mean.detach().cpu().numpy()

    
    ########################
    ### ERA5 Test Factual ###
    ########################
    
    # assign arrays
    x_era5_fact_reduced_mean_numpy = np.load(args.era5_fact_truth_save_path, mmap_mode="r") #x_era5_fact_reduced_mean.detach().cpu().numpy()

    
    ###############################
    ### ERA5 Test Counterfactual ###
    ###############################
    
    # assign arrays
    x_era5_cf_reduced_mean_numpy = np.load(args.era5_cf_truth_save_path, mmap_mode="r") #x_era5_cf_reduced_mean.detach().cpu().numpy()
    

    ####################
    ### DAE Datasets ###
    ####################
    ###########################
    ### Validation Ensemble ###
    ###########################
    dpa_val_ensemble = xr.open_dataset(f"{args.model_path}/{args.model}/validation_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_{args.eval_no_epochs}_epochs/eth_ensemble_after_{args.eval_no_epochs}_epochs/raw_ETH_gen_dpa_ens_{args.eval_no_epochs}_dataset.nc")
    
    
    # dpa spatial mean
    dpa_ensemble_spat_mean = dpa_val_ensemble.TREFHT.mean(dim="lat_x_lon")
    
    # ensemble mean
    dpa_ensemble_mean_spat_mean = dpa_ensemble_spat_mean.mean(dim="ensemble_member")
    #dpa = dpa_ensemble_mean_spat_mean.values
    
    ############################
    ### ETH Factual Ensemble ###
    ############################
    #dpa_eth_fact_ensemble = xr.open_dataset(f"{model_path}/{model}/.nc")
    dpa_eth_fact_ensemble = xr.open_dataset(f"{args.model_path}/{args.model}/eth_test_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_{args.eval_no_epochs}_epochs/eth_ensemble_after_{args.eval_no_epochs}_epochs/raw_ETH_gen_dpa_ens_{args.eval_no_epochs}_dataset.nc")
    
    
    # dpa spatial mean
    dpa_eth_fact_ensemble_spat_mean = dpa_eth_fact_ensemble.TREFHT.mean(dim="lat_x_lon")
    
    # ensemble mean
    dpa_eth_fact_ensemble_mean_spat_mean = dpa_eth_fact_ensemble_spat_mean.mean(dim="ensemble_member")
    #dpa = dpa_ensemble_mean_spat_mean.values
    
    #######################
    ### ETH CF Ensemble ###
    #######################
    #dpa_eth_cf_ensemble = xr.open_dataset(f"{model_path}/{model}/.nc")
    dpa_eth_cf_ensemble = xr.open_dataset(f"{args.model_path}/{args.model}/eth_test_set_reference_period_1950-1980_v6_dpa_train_settings.json/dpa_ensemble_after_{args.eval_no_epochs}_epochs/eth_ensemble_after_{args.eval_no_epochs}_epochs/raw_ETH_cf_gen_dpa_ens_{args.eval_no_epochs}_dataset.nc")
    
    # dpa spatial mean
    dpa_eth_cf_ensemble_spat_mean = dpa_eth_cf_ensemble.TREFHT.mean(dim="lat_x_lon")
    
    # ensemble mean
    dpa_eth_cf_ensemble_mean_spat_mean = dpa_eth_cf_ensemble_spat_mean.mean(dim="ensemble_member")
    
    #############################
    ### ERA5 Factual Ensemble ###
    #############################
    #dpa_era5_fact_ensemble = xr.open_dataset(f"{model_path}/{model}/.nc")
    
    # dpa spatial mean
    #dpa_era5_fact_ensemble_spat_mean = dpa_era5_fact_ensemble.TREFHT.mean(dim="lat_x_lon")
    
    # ensemble mean
    #dpa_era5_fact_ensemble_mean_spat_mean = dpa_era5_fact_ensemble_spat_mean.mean(dim="ensemble_member")
    #dpa = dpa_ensemble_mean_spat_mean.values
    
    ########################
    ### ERA5 CF Ensemble ###
    ########################
    #dpa_era5_cf_ensemble = xr.open_dataset(f"{model_path}/{model}/.nc")
    
    # dpa spatial mean
    #dpa_era5_cf_ensemble_spat_mean = dpa_era5_cf_ensemble.TREFHT.mean(dim="lat_x_lon")
    
    # ensemble mean
    #dpa_era5_cf_ensemble_mean_spat_mean = dpa_era5_cf_ensemble_spat_mean.mean(dim="ensemble_member")

    ################    
    ### Evaluate ###
    ################
    test_sets = [test, x_eth_fact_reduced_mean_numpy, x_eth_cf_reduced_mean_numpy[:2*4769]]
    dpa_spat_means = [dpa_ensemble_spat_mean, dpa_eth_fact_ensemble_spat_mean, dpa_eth_cf_ensemble_spat_mean[:,:2*4769]]
    dae_ensemble_means_spat_means = [dpa_ensemble_mean_spat_mean, dpa_eth_fact_ensemble_mean_spat_mean, dpa_eth_cf_ensemble_mean_spat_mean[:2*4769]]
    comments = ["validation","eth_factual","eth_cf", "era5_factual", "era5_cf"]
    
    # test, dpa_ensemble_mean_spat_mean, dpa_ensemble_spat_mean
    
    for test_val, dae_ens_mean_spat_mean, dae_ensemble_spat_mean, save_comment in zip(test_sets, dae_ensemble_means_spat_means, dpa_spat_means, comments):
        compare_distrs(test_val, dae_ens_mean_spat_mean.values, dae_ensemble_spat_mean.values, save_comment)
    




    
def compare_distrs(test, dpa, dpa_ensemble_spat_mean, save_path, comment):
    x1 = dpa 
    x2 = test # assign loaded numpy array here
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram (density=True → normalized)
    ax.hist(x1, bins=40, density=True, alpha=0.5, label="DPA ensemble")
    ax.hist(x2, bins=40, density=True, alpha=0.5, label="CESM2 Validation Data")
    
    # Compute statistics
    mean1, std1 = np.mean(x1), np.std(x1)
    mean2, std2 = np.mean(x2), np.std(x2)
    
    print(f"{comment} mean bias: {np.abs(mean2 - mean1)}")
    print(f"{comment} set true std: {std2}")
    print(f"{comment} set DAE std: {std1}")
    
    # Add mean lines
    ax.axvline(mean1, linestyle="--", linewidth=2, color="tab:blue")
    ax.axvline(mean2, linestyle="--", linewidth=2, color ="tab:orange")
    
    # Add ±2σ lines
    ax.axvline(mean1 - 2*std1, linestyle=":", linewidth=2, color="tab:blue")
    ax.axvline(mean1 + 2*std1, linestyle=":", linewidth=2, color="tab:blue")
    
    ax.axvline(mean2 - 2*std2, linestyle=":", linewidth=2, color ="tab:orange")
    ax.axvline(mean2 + 2*std2, linestyle=":", linewidth=2, color ="tab:orange")
    
    ax.set_xlabel("Temperature anomaly")
    ax.set_ylabel("Density")
    
    custom_patch = Patch(
        facecolor='tab:orange',
        edgecolor='tab:blue'
    )
    
    # Get existing legend entries
    handles, labels = ax.get_legend_handles_labels()
    
    # Append your custom entry
    handles.append(custom_patch)
    labels.append(f"mean bias = {np.abs(mean2 - mean1)}")
    
    # Rebuild legend
    ax.legend(handles, labels)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    # save figure 
    #plt.savefig(f"{save_path}/{comment}_validation_T_distrs_comparison.pdf")
    plt.show()
    
    
    # Quantile-Quantile
    quantiles = np.arange(0.05,1,0.05)
    quantiles_dpa = np.quantile(dpa, quantiles)
    quantiles_test = np.quantile(test, quantiles)
    plt.scatter(quantiles_test, quantiles_dpa, marker='x')
    
    # Plot 1:1 line
    x = np.linspace(-5, 5, 100)
    plt.plot(x, x, color="black", linestyle="--")
    plt.xlabel("True quantiles")
    plt.ylabel("DAE quantiles")
    #plt.savefig(f"{save_path}/{comment}_QQ_T_distrs_comparison.pdf")
    plt.show()
    mae_qq = np.mean(np.abs(quantiles_test-quantiles_dpa))
    print(f"comment Q-Q MAE: {mae_qq}")
    
    
    # Coverage-Quantiles
    dpa_ensemble_spat_mean_quantiles = np.quantile(dpa_ensemble_spat_mean, quantiles, axis=0)
    print("DPA quantiles shape:", dpa_ensemble_spat_mean_quantiles.shape)
    cover_dpa = evaluation.compute_coverage_per_quantile(test, dpa_ensemble_spat_mean_quantiles.T, quantiles)
    plt.scatter(quantiles, cover_dpa, marker='x')
    plt.plot(x, x, color="black", linestyle="--")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel("Fraction of points in quantile")
    plt.xlabel("Nominal quantiles")
    #plt.savefig(f"{save_path}/{comment}_CQ_T_distrs_comparison.pdf")
    mae_cq = np.mean(np.abs(quantiles-cover_dpa))
    print(f"{comment} C-Q MAE: {mae_cq}")
    return 

if __name__ == "__main__":
    main()