import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sys
sys.path.append("..")
from dpa.dpa_fit import DPA
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

def main():

    import argparse
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DPA and save loss curves.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output figures")
    parser.add_argument("--training_epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save log file")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # First, transpose to (time, lat, lon) if needed
    def data_to_torch(ds):
        temp_data = ds['Temperature']
        data = temp_data.transpose('time', 'lat', 'lon')

        # Now convert to numpy
        data_np = data.values  # Shape: (time, lat, lon)

        # Flatten lat and lon together
        time_steps, lat_dim, lon_dim = data_np.shape
        data_np = data_np.reshape(time_steps, lat_dim * lon_dim)


        data_np = data_np  # Shape: (grid_cell, timestep)

        # Finally, convert to torch tensor
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        print(data_tensor.shape)
        return data_tensor




    # load my temperature data
    # Load your NetCDF file
    ds_train = xr.open_dataset("/work/fl53wumy-llaae_data_new/llaae_data/ds_le.nc")
    ds_test = xr.open_dataset("/work/fl53wumy-llaae_data_new/llaae_data/ds_eth_test.nc")

    # transform to torch tensors
    x_tr = data_to_torch(ds_train)
    x_te = data_to_torch(ds_test)

    # initilize model
    dpa = DPA(data_dim=1024, latent_dims=[10,9,8,7,6,5,4,3,2,1,0], num_layer=4, hidden_dim=500, device=device)
    print("Now starting training ...")
    dpa.train(x_tr, x_te, batch_size=500, num_epochs=args.training_epochs, save_model_every=5, print_every_nepoch=1, save_dir=args.save_dir, save_loss=True)
    print("Training completed")
    def parse_log_file(log_file_path):
        """
        Parse the .log file to extract training and testing losses.

        Args:
            log_file_path (str): Path to the .log file.

        Returns:
            dict: A dictionary containing train and test losses over epochs.
        """
        train_losses = []
        test_losses = []
        with open(log_file_path, 'r') as file:
            for line in file:
                # Match train loss lines
                train_match = re.match(r"\[Epoch \d+\] (.+)", line)
                if train_match:
                    train_losses.append([float(x.strip()) for x in train_match.group(1).split(",")])

                # Match test loss lines
                test_match = re.match(r"\(test\)\s+(.+)", line)
                if test_match:
                    test_losses.append([float(x.strip()) for x in test_match.group(1).split(",")])

        return {
            "train": np.array(train_losses),
            "test": np.array(test_losses),
        }

    def plot_loss_evolution(loss_data, k_levels=None, save_path=None):
        """
        Plot the loss evolution over epochs for selected `k` levels during training and testing.

        Args:
            loss_data (dict): A dictionary containing train and test losses over epochs.
            k_levels (list, optional): List of `k` levels to plot. Defaults to all available levels.
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        train_losses = loss_data['train']
        test_losses = loss_data['test']

        if k_levels is None:
            k_levels = range(train_losses.shape[1])  # Default to all levels

        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 6))
        for k in k_levels:
            if k < train_losses.shape[1]:  # Ensure k is valid
                plt.plot(epochs, train_losses[:, k], label=f"Train Loss Level {k}")
                if len(test_losses) > 0:
                    plt.plot(epochs, test_losses[:, k], '--', label=f"Test Loss Level {k}")
            else:
                print(f"Warning: k level {k} is out of bounds and will be ignored.")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Evolution Over Epochs")
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))

    # Example usage
    log_file_path = f"{args.save_dir}/loss_log.txt"  # Replace with the path to your .log file
    loss_data = parse_log_file(log_file_path)

    # Specify the `k` levels you want to plot, or set to `None` to plot all levels
    k_levels_to_plot = [10]  # Example: Plot only levels 0 and 2
    plot_loss_evolution(loss_data, k_levels=k_levels_to_plot, save_path=f"{args.output_dir}/loss_evolution_k_levels.png")  # Replace with None to display the plot


if __name__ == "__main__":
    main()
