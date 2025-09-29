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
    x_entire_reduced, mask_x_entire = ut.remove_nan_columns(x_entire)

    print("Mask entire created")

    # load DPA predicted temperatures
    predicts_path = "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/eth_predictions/factual_predictions/"
    dpa_t_samples_raw, dpa_t_samples, t_arr_raw, t_arr_restored = ut.load_dpa_predicts(predicts_path, mask_x_entire, save_array=True, raw_hull=(103,14307,648), restored_hull=(103,14307,1024))

    print("Now saving arrays ...")
    torch.save(t_arr_raw, "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/eth_predictions/factual_predictions/dpa_ens_97_arr_raw.pt")
    print("raw array saved")
    torch.save(t_arr_restored, "/work/fl53wumy-llaae_data_new/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/eth_predictions/factual_predictions/dpa_ens_97_arr_restored.pt")
    print("restored array saved")

    



if __name__ == "__main__":
    main()