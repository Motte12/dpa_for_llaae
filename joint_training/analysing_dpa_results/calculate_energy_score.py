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

    # set train/test split
    ds_test = ds.isel(time=slice(-64000, 476900)) #4769 * 80
    
    # transform to torch tensors
    x_te = ut.data_to_torch(ds_test, "TREFHT")
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)

    print("Now loading DPA ensemble")
    # load DPA predicted temperatures
    predicts_path = "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/"
    dpa_t_samples_raw, dpa_t_samples, t_arr_raw, t_arr_restored = ut.load_dpa_predicts(predicts_path, mask_x_te, save_array=False)

    print("Now calculating energy score")
    # caluclate energy score
    # e_loss = energy_loss(x_te_reduced, dpa_t_samples_raw)
    eloss_map = ut.energy_loss_per_dim_efficient(x_te_reduced, dpa_t_samples_raw)

    
    print("Energy score calculated. Now saving energy score to array.")
    torch.save(eloss_map, "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ens_97_energy_loss_map.pt")
    

if __name__ == "__main__":
    main()