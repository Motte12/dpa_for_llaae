import torch
import torch.nn as nn
import argparse
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/')
import dpa_ensemble as de

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file_path", type=str, default = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v5_dpa_train_settings_home.json")
    args = parser.parse_args()

    
    model_path = "/work2/fl53wumy-llaae_ws_new/fl53wumy-dpa_data/fl53wumy-llaae_data_new_22092025-1763346001/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/baseline_quantile_regression/v5_data_quantile_regression_ger_gradient_descent_2026-03-25_13-07/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load checkpoint
    ckpt = torch.load(f"{model_path}/checkpoint_epoch_100.pth", map_location=device, weights_only=False)
    
    # rebuild model
    model = LinearMultiQuantileRegressor(
        n_features=ckpt["n_features"],
        n_cells=ckpt["n_cells"],
        n_quantiles=ckpt["n_quantiles"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    ##########################
    ### Load ETH test data ###
    ##########################
    z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf, mean_gmt, std_gmt = de.load_eth_test_data(args.settings_file_path, standardize_predictors=1)

    print("Predictors shape:", z500.shape)
    

    
    #############################
    ## End Load ETH test data ###
    #############################
    
    
    with torch.no_grad():
        preds = model(z500)
    
    print("predictions shape:", preds.shape)  # (100, 648, 3)

if __name__=="__main__":
    main()