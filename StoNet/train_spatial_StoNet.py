import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torchvision.utils import make_grid
from engression.models import StoNet, StoLayer
from engression import engression
from engression.loss_func import energy_loss_two_sample
import argparse
import json
import xarray as xr
import torch.nn as nn
from sklearn.manifold import TSNE
import pickle


import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut

def main():
    #######################
    ### Input Arguments ###
    #######################
    parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")

    ## Logistics
    #parser.add_argument('--save_dir', type=string, default= , help='Save directory')
    parser.add_argument('--settings_file', type=str, help='Settings file')

    ## Model
    parser.add_argument('--encoder', type=str, default='learnable', help='Use learnable encoder (=neural) or fixed (=PCA)')
    parser.add_argument('--in_dim', type=int, help='Input dimension into encoder')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension of the DPA')
    parser.add_argument('--num_layer', type=int, help='Number of layers in encoder and decoder')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dims in encoder and decoder')
    parser.add_argument('--noise_dim_dec', type=int, help='Dimension of noise added to decoder')
    parser.add_argument('--out_activation', type=str, help='Output activation')
    parser.add_argument('--resblock', type=int, default=1, help="Whether to use residual block")
    
    parser.add_argument('--in_dim_lm', type=int, default=1001, help="Input dimension of the latent map")
    parser.add_argument('--noise_dim_lm', type=int, help='Dimension of noise added in latent map')
    parser.add_argument('--num_layer_lm', type=int, help='Number of layers in latent map')
    parser.add_argument('--hidden_dim_lm', type=int, help='Hidden dims in latent map')

    ## Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_norm', type=int, default=0, help='Whether to use batch normalisation')
    parser.add_argument('--save_dir', type=str, default=None, help="Where to save output.")

    ## Loss
    parser.add_argument('--lam', type=float, default=1.0, help="Weight between energy loss in Y and latent space")
    #parser.add_argument('--include_KL', type=bool, default="False", help='Whether to include KL loss')

    # Parse arguments
    args = parser.parse_args()
    settings_file_path = args.settings_file
    print("settings_file:", settings_file_path)
    print("args:", args)

    encoder = args.encoder
    in_dim = args.in_dim
    latent_dim = args.latent_dim
    num_layers = args.num_layer
    hidden_dim = args.hidden_dim
    noise_dim_dec = args.noise_dim_dec
    out_act = None #args.out_activation
    resblock = bool(args.resblock)
    print("resblock:", resblock)

    in_dim_lm = args.in_dim_lm
    noise_dim_lm = args.noise_dim_lm
    num_layers_lm = args.num_layer_lm
    hidden_dim_lm = args.hidden_dim_lm
    beta=1

    # train settings
    bn = bool(args.batch_norm)
    print("bn:", bn)
    batch_size=args.batch_size
    #include_KL = args.include_KL
    lam = args.lam
    lr = args.lr
    training_epochs=args.epochs

    
    
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    
    # create save directory
    if args.save_dir is not None:  
        save_dir = f"{args.save_dir}_{num_layers}_{hidden_dim}_{noise_dim_dec}_bs{batch_size}_bnis{bn}_{training_epochs}_epochs_trained/"
    else:
        save_dir = f"{settings['output_dir']}_{num_layers}_{hidden_dim}_{noise_dim_dec}_bs{batch_size}_bnis{bn}_{training_epochs}_epochs_trained/"
    
    # create directory
    os.makedirs(save_dir, exist_ok=True)
    print("save_directory:", save_dir)

    # save this script for logging info
    # Path to this script
    current_script = os.path.realpath(__file__)
    
    # Copy the script
    shutil.copy(current_script, f"{save_dir}used_training_script.py")

    
    #############
    ### Seeds ###
    #############
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    ##################
    ### Cuda Setup ###
    ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    

    #################
    ### Load Data ###
    #################

    # change here
    
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

    z500_eth = xr.open_dataset(settings['dataset_z500_eth_test'])
    predictors_combined_eth = z500_eth.pseudo_pcs.values 

    ## train
    train_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
    X_torch = torch.from_numpy(train_predictors)

    ## validation
    validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :])
    X_val_torch = torch.from_numpy(validation_predictors)

    ## test
    test_predictors, _, _ = ut.standardize_numpy(predictors_combined_eth)
    X_test_torch = torch.from_numpy(test_predictors)



    ###
    #ds_z500_pre = xr.open_dataset(settings['dataset_z500'])

    # Z500 not standardized yet
    #ds_z500, _, _ = ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values) 

    # Z500 already standardized
    #ds_z500 = ds_z500_pre.pseudo_pcs.values #ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values)
    #print("z500 shape", ds_z500.shape)
    #z500 = torch.from_numpy(ds_z500)
    #print("z500 shape", z500.shape)
    
    
    #z500_train = z500[:int(4769 * 90),:]
    #z500_test = z500[int(4769 * 90):,:]

    # remove NaNs from Temperature data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)

    # create data loader Temperature
    #train_dataset = TensorDataset(X_torch, x_tr_reduced)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    #print(f"Number of batches: {len(train_loader)}")
    
    # create test loader Temperature
    #test_dataset = TensorDataset(X_val_torch, x_te_reduced) # test is actually validation
    #test_loader_in = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #print(f"Number of batches: {len(test_loader_in)}")

    # write all args to a json file (for logs)
    log_file_settings_name = os.path.join(save_dir, 'model_and_train_settings.json')
    with open(log_file_settings_name, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    
    ##############
    ### StoNet ###
    ##############

    # Fit an engression model
    engressor = engression(X_torch, 
                       x_tr_reduced, 
                       resblock=resblock, 
                       num_layer=num_layers, 
                       hidden_dim=hidden_dim, 
                       noise_dim=noise_dim_dec, 
                       add_bn = bn,
                       lr=lr, 
                       num_epochs=training_epochs, 
                       standardize=False,
                       batch_size=batch_size,
                       print_every_nepoch=1, 
                       device=device)

    # save model
    with open(f"{save_dir}{training_epochs}_epochs_trained_engressor.pkl", "wb") as f:
        pickle.dump(engressor, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    # create test ensemble
    y_pred_standardized = engressor.sample(X_test_torch, sample_size=100)

    # save test ensemble
    torch.save(y_pred_standardized, f"{save_dir}{training_epochs}_epochs_trained_engressor_predicted_ensemble.pt")



if __name__ == "__main__":
    main()
