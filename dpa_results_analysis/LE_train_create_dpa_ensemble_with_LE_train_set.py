###########################
### Create DPA ensemble ###
###########################
## 1) load test data
## 2) load encoder, decoder, latent map
## 3) create ensemble
## 4) create .nc array containing results

import torch
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss, energy_loss_two_sample

import xarray as xr
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.manifold import TSNE
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil
import argparse

import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import dpa_ensemble as de
import utils as ut
import evaluation

def main():
    parser = argparse.ArgumentParser(description="Example script with arguments")

    # Define arguments
    parser.add_argument("--autoencode_only", type=int, default=0)
    parser.add_argument("--ens_members", type=int, help="Number of ensemble members to create")
    parser.add_argument("--save_path_ensemble_single", type=str, help="path for saving single ens members")
    parser.add_argument("--model_path", type=str, help="path to DPA model components")
    parser.add_argument("--encoder_model", type=str, help="encoder filename")
    parser.add_argument("--decoder_model", type=str, help="decoder filename")
    parser.add_argument("--latent_map_model", type=str, help="latent_map filename")
    parser.add_argument("--no_epochs", type=int, help="Number of epochs")
    parser.add_argument("--settings_file_path", type=str, help="Settings file path.")


      # --- Encoder and model structure ---
    parser.add_argument("--encoder", type=str, default="learnable",
                        choices=["learnable", "PCA"],
                        help="Type of encoder architecture to use (default: learnable).")

    parser.add_argument("--latent_dim", type=int, default=50,
                        help="Dimensionality of latent space (default: 50).")

    parser.add_argument("--in_dim", type=int, default=648,
                        help="Input dimension size (default: 648).")

    parser.add_argument("--hidden_dim", type=int, default=50,
                        help="Hidden layer dimensionality (default: 50).")

    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of layers in the encoder/decoder (default: 6).")

    parser.add_argument("--noise_dim_dec", type=int, default=5,
                        help="Noise dimension for decoder (default: 5).")

    # --- Latent model / latent mapping (LM) ---
    parser.add_argument("--in_dim_lm", type=int, default=1001,
                        help="Input dimension for latent model (default: 1001).")

    parser.add_argument("--num_layers_lm", type=int, default=2,
                        help="Number of layers in latent model (default: 2).")

    parser.add_argument("--hidden_dim_lm", type=int, default=50,
                        help="Hidden dimension in latent model (default: 50).")

    parser.add_argument("--noise_dim_lm", type=int, default=20,
                        help="Noise dimension for latent model (default: 20).")

    # --- Activation and normalization ---
    parser.add_argument("--out_act", type=str, default=None,
                        choices=[None, "relu", "tanh", "sigmoid"],
                        help="Output activation function (default: None).")

    parser.add_argument("--resblock", action="store_true", default=True,
                        help="Use residual blocks (default: True).")

    parser.add_argument("--no_resblock", dest="resblock", action="store_false",
                        help="Disable residual blocks.")

    parser.add_argument("--bn", action="store_true", default=True,
                        help="Use batch normalization (default: True).")

    parser.add_argument("--no_bn", dest="bn", action="store_false",
                        help="Disable batch normalization.")

    # --- Regularization and training ---
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Lambda regularization coefficient (default: 0.5).")

    parser.add_argument("--bs", type=int, default=128,
                        help="Batch size for training (default: 128).")
          
    
    

    args = parser.parse_args()
               
    ens_members = args.ens_members #100
    save_path_ensemble_single = f"{args.save_path_ensemble_single}train_set_ensemble_after_{args.no_epochs}_epochs" #"/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_100epochs/train_set_ensemble_after_100_epochs"
    os.makedirs(save_path_ensemble_single, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    
    
    # LOAD DPA MODEL
    # set model parameters
    encoder=args.encoder #"learnable"
    latent_dim=args.latent_dim #50
    in_dim=args.in_dim #648
    hidden_dim=args.hidden_dim #50
    num_layers=args.num_layers #6
    noise_dim_dec=args.noise_dim_dec #5
    
    in_dim_lm=args.in_dim_lm #1001
    num_layers_lm=args.num_layers_lm #2
    hidden_dim_lm=args.hidden_dim #50
    noise_dim_lm=args.noise_dim_lm #20

    out_act = args.out_act #None
    resblock=args.resblock #True
    
    bn= args.bn #True
    lambd=args.lambd #0.5
    bs=args.bs #128

    # create_ensemble() saves ensemble and return mask
    mask, ds_train, ds_test, x_te_reduced = de.create_ensemble(ensemble_type="LE",
                                    ensemble_size=ens_members,
                                    save_path=save_path_ensemble_single,
                                    device=device,
                                    encoder=encoder,
                                    in_dim=in_dim,
                                    latent_dim=latent_dim,
                                    num_layers=num_layers,
                                    hidden_dim=hidden_dim,
                                    bn=bn,
                                    out_act=out_act,
                                    resblock=resblock, 
                                    noise_dim_dec=noise_dim_dec,
                                    in_dim_lm=in_dim_lm,
                                    num_layers_lm=num_layers_lm,
                                    hidden_dim_lm=hidden_dim,
                                    noise_dim_lm=noise_dim_lm,
                                    encoder_path=f"{args.model_path}/{args.encoder_model}",
                                    decoder_path=f"{args.model_path}/{args.decoder_model}",
                                    lm_path=f"{args.model_path}/{args.latent_map_model}",
                                    settings_file_path = args.settings_file_path, #"/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder/joint_training/v2_dpa_train_settings.json",
                                    create_train_ensemble=True,
                                    autoencode=args.autoencode_only
                                    )

    # save data to netCDF dataset
    # could replace with load_both_dpa_arrays
    tensor_list, tensor_list_raw, stacked, stacked_reshaped, ds = ut.load_dpa_arrays(path=f"{save_path_ensemble_single}/",
                                                                    mask=mask,
                                                                    ds_coords=ds_train,
                                                                    ens_members=ens_members,
                                                                    save_path=f"{save_path_ensemble_single}",
                                                                    no_epochs=args.no_epochs
                                                                    )

if __name__ == "__main__":
    main()