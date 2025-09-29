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

import sys
sys.path.append('/home/sc.uni-leipzig.de/fl53wumy/llaae_new/DistributionalPrincipalAutoencoder')
import dpa_ensemble as de
import utils as ut
import evaluation

def main_1():
    ens_members = 100
    save_path_ensemble_single = "/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/dpa_ensemble_after_30epochs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    
    
    # LOAD DPA MODEL
    # set model parameters
    encoder="learnable"
    latent_dim=50
    in_dim=648
    hidden_dim=50
    num_layers=6
    noise_dim_dec=5
    
    in_dim_lm=1001
    num_layers_lm=2
    hidden_dim_lm=50
    noise_dim_lm=20

    out_act = None
    resblock=True
    
    bn=True #False
    lambd=0.5
    bs=128
    #epochs=300

    # create_ensemble() saves ensemble and return mask
    mask, ds_test, x_te_reduced = de.create_ensemble(ensemble_size=ens_members,
                                    save_path=save_path_ensemble_single,
                                    device=device,
                                    encoder="learnable",
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
                                    encoder_path="/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/model_enc_30.pt",
                                    decoder_path="/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/model_dec_30.pt",
                                    lm_path="/work/fl53wumy-llaae_data_new_22092025/fl53wumy-llaae_data_new-1758244802/fl53wumy-llaae_data_new-1748049607/dpa_output/dpa_model3_tuning1/_50_6_50_5_1001_20_2_50_encoderislearnable_lambda0.5_bs128/model_pred_30.pt",
                                    create_counterfactual_ensemble=True
                                    )

    # save data to netCDF dataset
    # could replace with load_both_dpa_arrays
    tensor_list, stacked, stacked_reshaped, ds = ut.load_dpa_arrays(path=f"{save_path_ensemble_single}/",
                                                                    mask=mask,
                                                                    ds_coords=ds_test,
                                                                    ens_members=ens_members,
                                                                    save_path=f"{save_path_ensemble_single}")