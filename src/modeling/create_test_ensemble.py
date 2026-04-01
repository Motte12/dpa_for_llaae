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
sys.path.append('../utils')
import dpa_ensemble as de
import utils as ut
#import evaluation

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

    parser.add_argument("--noise_dim_dec", type=int, default=5, # was set to 5 before
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

    parser.add_argument("--bn", type=int, default=0,
                        help="Use batch normalization (default: 0).")

    # --- Regularization and training ---
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Lambda regularization coefficient (default: 0.5).")

    parser.add_argument("--bs", type=int, default=128,
                        help="Batch size for training (default: 128).")
    parser.add_argument("--standardize_predictors", type=int, default=0,
                        help="Whether to manually standardize predictors (only if raw data not standardized already).")
          
    
    

    args = parser.parse_args()
    print("Script started ...")

    if args.autoencode_only:
        save_path_ensemble_single = f"{args.save_path_ensemble_single}eth_ensemble_after_{args.no_epochs}_epochs"

    else:
        save_path_ensemble_single = f"{args.save_path_ensemble_single}eth_ensemble_after_{args.no_epochs}_epochs"
    
    os.makedirs(save_path_ensemble_single, exist_ok=True)
    print("save_path_ensemble_single", save_path_ensemble_single)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # create_ensemble() saves ensemble and return mask
    mask, ds_train, ds_test, x_te_reduced = de.create_ensemble(ensemble_type="ETH", #"ETH"
                                    ensemble_size=args.ens_members,
                                    save_path=save_path_ensemble_single,
                                    device=device,
                                    encoder=args.encoder,
                                    in_dim=args.in_dim,
                                    latent_dim=args.latent_dim,
                                    num_layers=args.num_layers,
                                    hidden_dim=args.hidden_dim,
                                    bn=args.bn,
                                    out_act=args.out_act,
                                    resblock=args.resblock, 
                                    noise_dim_dec=args.noise_dim_dec,
                                    in_dim_lm=args.in_dim_lm,
                                    num_layers_lm=args.num_layers_lm,
                                    hidden_dim_lm=args.hidden_dim_lm,
                                    noise_dim_lm=args.noise_dim_lm,
                                    encoder_path=f"{args.model_path}/{args.encoder_model}",
                                    decoder_path=f"{args.model_path}/{args.decoder_model}",
                                    lm_path=f"{args.model_path}/{args.latent_map_model}",
                                    settings_file_path=args.settings_file_path,
                                    create_factual_ensemble=True,
                                    create_counterfactual_ensemble=True,
                                    autoencode=args.autoencode_only,
                                    standardize_predictors=args.standardize_predictors
                                    )
    print("Ensemble created")
    # save data to netCDF dataset
    tensor_list, tensor_list_raw, stacked, stacked_reshaped, ds = ut.load_both_dpa_arrays(path=f"{save_path_ensemble_single}/",
                                                                    mask=mask,
                                                                    ds_coords=ds_test,
                                                                    ens_members=args.ens_members,
                                                                    save_path=f"{save_path_ensemble_single}",
                                                                    no_epochs=args.no_epochs,
                                                                    climate_list=["gen", "cf_gen"],
                                                                    )

if __name__ == "__main__":
    main()