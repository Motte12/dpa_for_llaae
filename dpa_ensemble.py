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
import utils as ut

def load_test_data(settings_file_path, standardize_predictors=0):
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)

    ### Load temperature data ###
    ds = xr.open_dataset(settings['dataset_trefht'])
    
    # set train/test split
    ds_train = ds.isel(time=slice(0, 20*4769)) # larger than 20 will likely get too data heavy
    ds_test = ds.isel(time=slice(90*4769, 476900))
    
    # transform to torch tensors
    x_tr = ut.data_to_torch(ds_train, "TREFHT")
    x_te = ut.data_to_torch(ds_test, "TREFHT")

    # remove NaNs from data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)

    ### Load Z500 data ###
    ds_z500_pre = xr.open_dataset(settings['dataset_z500'])
    if standardize_predictors:
        ds_z500, _, _ = ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values) # data is already standardized
    else:
        ds_z500 = ds_z500_pre.pseudo_pcs.values
    print("z500 dataset shape", ds_z500.shape)
    z500 = torch.from_numpy(ds_z500)
    print("z500 shape", z500.shape)
    
    
    z500_train = z500[0:int(4769 * 20),:]
    z500_test = z500[int(90*4769):476900,:]
    
    return z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced

def load_eth_test_data(settings_file_path, standardize_predictors=0):
    
    # TREFHT 
    ## factual
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    
    ### Load temperature data ###
    ds_test_eth_fact = xr.open_dataset(settings['dataset_trefht_eth_transient'])
    
    
    # transform to torch tensors
    x_te_eth_fact = ut.data_to_torch(ds_test_eth_fact, "TREFHT")

    # remove NaNs from data
    x_te_reduced_eth_fact, mask_x_te_eth_fact = ut.remove_nan_columns(x_te_eth_fact)
    

    ## counterfactual
    
    ### Load temperature data ###
    ds_test_eth_cf = xr.open_dataset(settings['dataset_trefht_eth_nudged_shifted'])
    
    
    # transform to torch tensors
    x_te_eth_cf = ut.data_to_torch(ds_test_eth_cf, "TREFHT")

    # remove NaNs from data
    x_te_reduced_eth_cf, mask_x_te_eth_cf = ut.remove_nan_columns(x_te_eth_cf)
    
    
    # Z500
    
    ### Load Z500 data ###
    ds_z500_pre = xr.open_dataset(settings['dataset_z500_eth_test'])

    print("ATTENTION: Z500 PC time-series is standardized manually here")
    if standardize_predictors:
        ds_z500_standardized, mean, std = ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values)
    else:
        ds_z500_standardized = ds_z500_pre.pseudo_pcs.values
    print("z500 dataset shape", ds_z500_standardized.shape)
    z500 = torch.from_numpy(ds_z500_standardized)

    return z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf 

    

def create_dpa_model(device,
                     encoder,
                     in_dim,
                     latent_dim,
                     num_layers,
                     hidden_dim,
                     bn,
                     out_act,
                     resblock, 
                     noise_dim_dec,
                     in_dim_lm,
                     num_layers_lm,
                     hidden_dim_lm,
                     noise_dim_lm,
                     encoder_path,
                     decoder_path,
                     lm_path
                     ):
    # Encoder
    if encoder == "PCA":
        pca, pcs = pcae.get_PC(ds_train)
        eofs_torch_first = pca.components_[:latent_dim, :] # take only the first latent_dim PCs, pca.components_ has shape: (n_components_total, n_features) = (modes, space)
        
        # "encode" by projecting onto PCs
        def model_enc(x, eofs=eofs_torch_first):
            teofs = ((torch.from_numpy(eofs)).T).to(device)
            return torch.matmul(x, teofs)
            
        
    elif encoder == "learnable":
        model_enc = StoNet(in_dim=in_dim,
                           out_dim=latent_dim,
                           num_layer=num_layers,
                           hidden_dim=hidden_dim,
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
    model_pred = StoNet(in_dim=in_dim_lm,
                        out_dim=latent_dim, 
                        num_layer=num_layers_lm,
                        hidden_dim=hidden_dim_lm, 
                        noise_dim=noise_dim_lm,
                        add_bn=bn, 
                        out_act=out_act, 
                        resblock=resblock).to(device)
    print("device:", device)
    # Load the state dict from the .pt file
    if str(device)=="cuda":
        # using GPU
        enc_dict = torch.load(encoder_path)
        dec_dict = torch.load(decoder_path)
        lm_dict = torch.load(lm_path)
    
    elif str(device)=="cpu":    
        # using CPU
        enc_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        dec_dict = torch.load(decoder_path, map_location=torch.device('cpu'))
        lm_dict = torch.load(lm_path, map_location=torch.device('cpu'))
    
    # Load the state dict into the internal DPA model
    model_enc.load_state_dict(enc_dict)
    model_dec.load_state_dict(dec_dict)
    model_pred.load_state_dict(lm_dict)

    print("DPA model created")

    total_params0 = sum(p.numel() for p in model_enc.parameters())
    print(f"Total encoder parameters: {total_params0:,}")

    total_params1 = sum(p.numel() for p in model_dec.parameters())
    print(f"Total decoder parameters: {total_params1:,}")

    total_params2 = sum(p.numel() for p in model_pred.parameters())
    print(f"Total latent map parameters: {total_params2:,}")
    
    
    return model_enc, model_dec, model_pred

def create_ensemble(ensemble_type,
                    ensemble_size,
                    save_path,
                    device,
                    encoder,
                    in_dim,
                    latent_dim,
                    num_layers,
                    hidden_dim,
                    bn,
                    out_act,
                    resblock, 
                    noise_dim_dec,
                    in_dim_lm,
                    num_layers_lm,
                    hidden_dim_lm,
                    noise_dim_lm,
                    encoder_path,
                    decoder_path,
                    lm_path,
                    settings_file_path,
                    create_factual_ensemble=False,
                    create_counterfactual_ensemble=False,
                    create_train_ensemble=False,
                    autoencode=False, # whether to use only autoencoder or not,
                    standardize_predictors=0
                   ):
    
    print("Autoencode:", autoencode)
    # load data
    if ensemble_type == "LE":
        #z500_test, z500_train, mask, ds_train, ds_test, x_te_reduced = load_test_data()
        z500_test, z500_train, mask, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced = load_test_data(settings_file_path, standardize_predictors)

    elif ensemble_type == "ETH":
        z500_test, mask, ds_test, ds_test_eth_cf, x_te_reduced, x_te_reduced_cf = load_eth_test_data(settings_file_path, standardize_predictors)
        #z500_test, mask, ds_test, x_te_reduced, x_te_reduced_cf = load_eth_test_data() # x_te_reduced is x_te_reduced_eth_fact
        
        # z500_standardized, mask_x_te_eth_fact, ds_test_eth_fact, x_te_reduced_eth_fact, x_te_reduced_eth_cf
    print("Data loaded")
    # create model
    model_enc, model_dec, model_pred = create_dpa_model(device,
                                                        encoder,
                                                        in_dim,
                                                        latent_dim,
                                                        num_layers,
                                                        hidden_dim,
                                                        bn,
                                                        out_act,
                                                        resblock, 
                                                        noise_dim_dec,
                                                        in_dim_lm,
                                                        num_layers_lm,
                                                        hidden_dim_lm,
                                                        noise_dim_lm,
                                                        encoder_path,
                                                        decoder_path,
                                                        lm_path
                                                       )
    if encoder == "learnable":
        model_enc.eval()
    model_dec.eval()
    model_pred.eval()

    # actually create ensemble
    if create_factual_ensemble:
        if autoencode:
            print("Autoencoding factual ETH test set ...")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_enc(x_te_reduced.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")
        else:
            print("Normal predictions ...")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_pred(z500_test.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")

    if create_train_ensemble:
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_pred(z500_train.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")

    if create_counterfactual_ensemble:
        if autoencode:
            print("Autoencoding nudged runs ... ")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_enc(x_te_reduced_cf.to(device).float()))
                torch.save(gen_te, f"{save_path}/cf_gen{i}_te.pt")
        else:
            print("Normal cf predictions ...")
            # replace GMTs with 0 for counterfactual predictions
            z500_test_cf = z500_test
            #z500_test_cf[:,-1] = 0 for fGMT not standardized
            z500_test_cf[:,-1] = -0.7389813694652794 # for fGMT standardized
            for i in range(1, ensemble_size+1):
                gen_te_cf = model_dec(model_pred(z500_test_cf.to(device).float()))
                torch.save(gen_te_cf, f"{save_path}/cf_gen{i}_te.pt")

    if ensemble_type == "ETH":
        ds_train = "no ds_train present"

    return mask, ds_train, ds_test, x_te_reduced


#############################
######### 1D ################
#############################
def create_dpa_model_1d(device,
                     encoder,
                     in_dim,
                     latent_dim,
                     num_layers,
                     hidden_dim,
                     bn,
                     out_act,
                     resblock, 
                     noise_dim_dec,
                     in_dim_lm,
                     num_layers_lm,
                     hidden_dim_lm,
                     noise_dim_lm,
                     encoder_path,
                     decoder_path,
                     lm_path
                     ):
    # Encoder
    if encoder == "PCA":
        pca, pcs = pcae.get_PC(ds_train)
        eofs_torch_first = pca.components_[:latent_dim, :] # take only the first latent_dim PCs, pca.components_ has shape: (n_components_total, n_features) = (modes, space)
        
        # "encode" by projecting onto PCs
        def model_enc(x, eofs=eofs_torch_first):
            teofs = ((torch.from_numpy(eofs)).T).to(device)
            return torch.matmul(x, teofs)
            
        
    elif encoder == "learnable":
        model_enc = StoNet(in_dim=in_dim,
                           out_dim=latent_dim,
                           num_layer=num_layers,
                           hidden_dim=hidden_dim,
                           noise_dim=0,
                           add_bn=bn,
                           out_act=out_act,
                           resblock=resblock).to(device)
    # Decoder
    model_dec = StoNet(in_dim=latent_dim,
                       out_dim=1, #in_dim,
                       num_layer=num_layers,
                       hidden_dim=hidden_dim,
                       noise_dim=noise_dim_dec,
                       add_bn=bn,
                       out_act=out_act,
                       resblock=resblock).to(device)
    # Latent Map
    model_pred = StoNet(in_dim=in_dim_lm,
                        out_dim=latent_dim, 
                        num_layer=num_layers_lm,
                        hidden_dim=hidden_dim_lm, 
                        noise_dim=noise_dim_lm,
                        add_bn=bn, 
                        out_act=out_act, 
                        resblock=resblock).to(device)
    print("device:", device)
    # Load the state dict from the .pt file
    if str(device)=="cuda":
        # using GPU
        enc_dict = torch.load(encoder_path)
        dec_dict = torch.load(decoder_path)
        lm_dict = torch.load(lm_path)
    
    elif str(device)=="cpu":    
        # using CPU
        enc_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
        dec_dict = torch.load(decoder_path, map_location=torch.device('cpu'))
        lm_dict = torch.load(lm_path, map_location=torch.device('cpu'))
    
    # Load the state dict into the internal DPA model
    model_enc.load_state_dict(enc_dict)
    model_dec.load_state_dict(dec_dict)
    model_pred.load_state_dict(lm_dict)

    print("DPA model created")

    total_params0 = sum(p.numel() for p in model_enc.parameters())
    print(f"Total encoder parameters: {total_params0:,}")

    total_params1 = sum(p.numel() for p in model_dec.parameters())
    print(f"Total decoder parameters: {total_params1:,}")

    total_params2 = sum(p.numel() for p in model_pred.parameters())
    print(f"Total latent map parameters: {total_params2:,}")
    
    
    return model_enc, model_dec, model_pred

def create_ensemble_1d(ensemble_type,
                    ensemble_size,
                    save_path,
                    device,
                    encoder,
                    in_dim,
                    latent_dim,
                    num_layers,
                    hidden_dim,
                    bn,
                    out_act,
                    resblock, 
                    noise_dim_dec,
                    in_dim_lm,
                    num_layers_lm,
                    hidden_dim_lm,
                    noise_dim_lm,
                    encoder_path,
                    decoder_path,
                    lm_path,
                    settings_file_path,
                    create_factual_ensemble=False,
                    create_counterfactual_ensemble=False,
                    create_train_ensemble=False,
                    autoencode=False, # whether to use only autoencoder or not,
                    standardize_predictors=0
                   ):
    
    print("Autoencode:", autoencode)
    # load data
    if ensemble_type == "LE":
        #z500_test, z500_train, mask, ds_train, ds_test, x_te_reduced = load_test_data()
        z500_test, z500_train, mask, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced = load_test_data(settings_file_path, standardize_predictors)

    elif ensemble_type == "ETH":
        z500_test, mask, ds_test, ds_test_eth_cf, x_te_reduced, x_te_reduced_cf = load_eth_test_data(settings_file_path, standardize_predictors)
        #z500_test, mask, ds_test, x_te_reduced, x_te_reduced_cf = load_eth_test_data() # x_te_reduced is x_te_reduced_eth_fact
        
        # z500_standardized, mask_x_te_eth_fact, ds_test_eth_fact, x_te_reduced_eth_fact, x_te_reduced_eth_cf
    print("Data loaded")
    # create model
    model_enc, model_dec, model_pred = create_dpa_model_1d(device,
                                                        encoder,
                                                        in_dim,
                                                        latent_dim,
                                                        num_layers,
                                                        hidden_dim,
                                                        bn,
                                                        out_act,
                                                        resblock, 
                                                        noise_dim_dec,
                                                        in_dim_lm,
                                                        num_layers_lm,
                                                        hidden_dim_lm,
                                                        noise_dim_lm,
                                                        encoder_path,
                                                        decoder_path,
                                                        lm_path
                                                       )
    if encoder == "learnable":
        model_enc.eval()
    model_dec.eval()
    model_pred.eval()

    # actually create ensemble
    if create_factual_ensemble:
        if autoencode:
            print("Autoencoding factual ETH test set ...")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_enc(x_te_reduced.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")
        else:
            print("Normal predictions ...")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_pred(z500_test.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")

    if create_train_ensemble:
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_pred(z500_train.to(device).float()))
                torch.save(gen_te, f"{save_path}/gen{i}_te.pt")

    if create_counterfactual_ensemble:
        if autoencode:
            print("Autoencoding nudged runs ... ")
            for i in range(1, ensemble_size+1):
                gen_te = model_dec(model_enc(x_te_reduced_cf.to(device).float()))
                torch.save(gen_te, f"{save_path}/cf_gen{i}_te.pt")
        else:
            print("Normal cf predictions ...")
            # replace GMTs with 0 for counterfactual predictions
            z500_test_cf = z500_test
            #z500_test_cf[:,-1] = 0 for fGMT not standardized
            z500_test_cf[:,-1] = -0.7389813694652794 # for fGMT standardized
            for i in range(1, ensemble_size+1):
                gen_te_cf = model_dec(model_pred(z500_test_cf.to(device).float()))
                torch.save(gen_te_cf, f"{save_path}/cf_gen{i}_te.pt")

    if ensemble_type == "ETH":
        ds_train = "no ds_train present"

    return mask, ds_train, ds_test, x_te_reduced

    
    
    
    
    