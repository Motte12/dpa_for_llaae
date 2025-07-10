import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torchvision.utils import make_grid
from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss_two_sample
import argparse
import json
import xarray as xr
import torch.nn as nn
from sklearn.manifold import TSNE

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import utils as ut

def main():
    #######################
    ### Input Arguments ###
    #######################
    parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")

    ## Logistics
    parser.add_argument('--save_dir', type=string, default=, help='Save directory')
    parser.add_argument('--settings_file', type=string, help='Save directory')


    
    ## Model
    parser.add_argument('--in_dim', type=int, help='Input dimension into encoder')
    parser.add_argument('--out_dim', type=int, help='Output dimension of encoder')
    parser.add_argument('--num_layer', type=int, help='Number of layers in encoder and decoder')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dims in encoder and decoder')
    parser.add_argument('--noise_dim', type=int, help='Dimension of noise added to decoder')
    parser.add_argument('--out_activation', type=str, help='Output activation')



    
    ## Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_norm', type=int, default=0, help='Whether to use batch normalisation')

    # Parse arguments
    args = parser.parse_args()
    save_dir = {args.save_dir}
    settings_file_path = {args.settings_file} #'dpa_train_settings.json'

    in_dim = {args.in_dim}
    out_dim = {args.out_dim}
    num_layers = {args.num_layer}
    hidden_dim = {args.hidden_dim}
    noise_dim = {args.noise_dim}

    bn = {args.batch_norm}
    
    


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

    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
      
    # Save to a new file for logging
    with open(f"used_settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    
    # Load temperature data
    ds = xr.open_dataset(settings['dataset_trefht'])
    print("Dataset:", settings['dataset_trefht'])

    # set train/test split
    ds_train = ds.isel(time=slice(0, 128000)) #4769 * 80
    ds_test = ds.isel(time=slice(-64000, 476900)) #4769 * 80
    
    # transform to torch tensors
    x_tr = ut.data_to_torch(ds_train, "TREFHT")
    x_te = ut.data_to_torch(ds_test, "TREFHT")
    
    # load Z500
    ds_z500_pre = xr.open_dataset(settings['dataset_z500'])
    ds_z500, _, _ = ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values)
    print("z500 shape", ds_z500.shape)
    z500 = torch.from_numpy(ds_z500)
    print("z500 shape", z500.shape)
    
    
    z500_train = z500[:128000,:]
    z500_test = z500[-64000:,:]

    # remove NaNs from data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)

    ###################
    ### Build Model ###
    ###################
    # model parameters 
    in_dim = x_tr_reduced.shape[1]
    latent_dim = {args.latent_dim} #50

    num_layers = {args.num_layers} #6
    num_layers = {args.num_layers_lm} #6

    hidden_dim = {args.hidden_dim} #100
    hidden_dim_latent_map = {args.hidden_dim_lm} #50

    noise_dim_dec= {args.noise_dim_dec} #20
    noise_dim_lm = {args.noise_dim_lm} #20 
    resblock = True
    out_act = None
    beta = {args.beta} #1
    
    # training parameters 
    batch_size = {args.batch_size}# 128 #190
    bn = False
    lr = {args.lr} #1e-4
    
    # Encoder
    model_enc = StoNet(in_dim=in_dim,
                       out_dim=latent_dim,
                       num_layer=num_layers,
                       hidden_dim=hidden_dim_latent_map,
                       noise_dim=0,
                       add_bn=bn,
                       out_act=out_act,
                       resblock=resblock).to(device)
    # Decoder
    model_dec = StoNet(in_dim=latent_dim,
                       out_dim=in_dim,
                       num_layer=num_layers,
                       hidden_dim=hidden_dim,
                       noise_dim=noise_dim_dec,
                       add_bn=bn,
                       out_act=out_act,
                       resblock=resblock).to(device)
    # Latent Map
    model_pred = StoNet(in_dim=1000,
                        out_dim=latent_dim, 
                        num_layer=num_layers_lm,
                        hidden_dim=hidden_dim_lm, 
                        noise_dim=noise_dim_lm,
                        add_bn=bn, 
                        out_act=out_act, 
                        resblock=resblock).to(device)
    
    
    optimizer = torch.optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_pred.parameters()), lr=lr)

    






















if __name__ == "__main__":
    main()