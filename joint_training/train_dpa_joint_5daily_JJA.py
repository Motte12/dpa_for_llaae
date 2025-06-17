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
import argparse
import os
import json
from datetime import datetime
#from dpa.dpa_fit import DPA
import dpa.dpa_fit as fit
import importlib


# First, transpose to (time, lat, lon) if needed
def data_to_torch(ds, variable):
    temp_data = ds[variable]
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

def predictors_to_torch(ds, variable):
    temp_data = ds[variable]
    data = temp_data.transpose('time', 'mode')
    
    # Now convert to numpy
    data_np = data.values  # Shape: (time, lat, lon)

    
    # Flatten lat and lon together
    #time_steps, lat_dim, lon_dim = data_np.shape
    #data_np = data_np.reshape(time_steps, lat_dim * lon_dim)
    
    
    #data_np = data_np  # Shape: (grid_cell, timestep)
    
    # Finally, convert to torch tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    print(data_tensor.shape)
    return data_tensor

def main():

    # print cuda specs
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA not available.")

    # read contents from the settings.json file
    settings_file_path = 'dpa_train_settings.json'

    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    
    # Get current date and time
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{settings['output_dir']}_jointDPA_{timestamp_str}/"

    # Create directories
    os.makedirs(output_dir, exist_ok=True)    

    # Save to a new file for logging
    with open(f"{output_dir}/used_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

    # load my temperature data
    # Load your NetCDF file
    ds = xr.open_dataset(settings['dataset_trefht'])

    ds_train = ds.isel(time=slice(0, 4769 * 80))
    ds_test = ds.isel(time=slice(4769 * 80, 476900))
    
    # transform to torch tensors
    x_tr = data_to_torch(ds_train, "TREFHT")
    x_te = data_to_torch(ds_test, "TREFHT")

    # load Z500
    ds_z500 = xr.open_dataset(settings['dataset_z500'])
    z500 = predictors_to_torch(ds_z500, "pseudo_pcs")
    z500_train = z500[:381520,:]
    z500_test = z500[381520:,:]

    # initilize model
    dpa = fit.DPA(data_dim=1024,
                  latent_dims=settings['latent_dims'],
                  num_layer=settings['num_layer'],
                  hidden_dim=settings['hidden_dim'],
                  device=device,
                  joint = True)
    
    print("Now starting training ...")
    
    dpa.train(x_tr, 
              z500_train, 
              x_te,
              z500_test, 
              batch_size=500, 
              num_epochs=settings['training_epochs'], 
              save_model_every=settings['save_model_every'], 
              print_every_nepoch=settings['print_every_epoch'], 
              save_dir=output_dir, 
              save_loss=True,
              print_all_k = True)

    
    print("Training completed")
    


if __name__ == "__main__":
    main()