import xarray as xr
import matplotlib.pyplot as plt
import importlib
import numpy as np
from engression.loss_func import energy_loss, energy_loss_two_sample
import torch
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut
device = torch.device("cpu")
print("device:", device)
import torch, gc
import pickle
import matplotlib.pyplot as plt
import json

def main():
    # LOAD ORIGINAL DATA

    settings_file_path = "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/dpa_train_settings.json"
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    
    ds = xr.open_dataset(settings['dataset_trefht'])
    x_entire = ut.data_to_torch(ds, "TREFHT")
    x_entire_reduced, mask = ut.remove_nan_columns(x_entire)
    ds_test = ds.isel(time=slice(-64000, 476900))


    print("Mask entire created")

    # load DPA predicted temperatures
    predicts_path = "/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/"
    tensor_list, stacked, stacked_reshaped, ds = ut.load_dpa_arrays(path=predicts_path, mask=mask, ds_coords=ds_test, save_path="/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ens_97_dataset_restored.zarr")

    print("Array saved")
    



if __name__ == "__main__":
    main()