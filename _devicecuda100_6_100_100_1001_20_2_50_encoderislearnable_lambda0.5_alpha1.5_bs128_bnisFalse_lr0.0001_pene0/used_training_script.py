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
import pca_encoder as pcae


import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil
import sys
sys.path.append('../utils')
import utils as ut

def main():

    ##############
    ### Parser ###
    ##############
    parser = argparse.ArgumentParser(description="Train a model with given hyperparameters.")
    parser.add_argument('--settings_file', type=str, default="../../settings.json", help='Settings file')
    args = parser.parse_args()

    # load default settings from settings.json
    # settings
    with open(args.settings_file, 'r') as file:
        settings = json.load(file)

    model_params = settings['model_parameters']
    train_params = settings['training_parameters']
    
    #####################
    ### add Arguments ###
    #####################

    ## Logistics

    ## Model
    parser.add_argument('--encoder', type=str, default=model_params['enc'], help='Use learnable encoder (=neural) or fixed (=PCA)')
    parser.add_argument('--in_dim', type=int, default=model_params['in_dim'], help='Input dimension into encoder')
    parser.add_argument('--latent_dim', type=int, default=model_params['ld'], help='Latent dimension of the DPA')
    parser.add_argument('--num_layer', type=int, default=model_params['nln'], help='Number of layers in encoder and decoder')
    parser.add_argument('--hidden_dim', type=int, default=model_params['hdn'], help='Hidden dims in encoder and decoder')
    parser.add_argument('--noise_dim_dec', type=int, default=model_params['ndd'], help='Dimension of noise added to decoder')
    parser.add_argument('--resblock', type=int, default=model_params['resblock'], help="Whether to use residual block")
    
    parser.add_argument('--in_dim_lm', type=int, default=model_params['in_dim_lm'], help="Input dimension of the latent map")
    parser.add_argument('--noise_dim_lm', type=int, default=model_params['ndl'], help='Dimension of noise added in latent map')
    parser.add_argument('--num_layer_lm', type=int, default=model_params['num_layer_lm'], help='Number of layers in latent map')
    parser.add_argument('--hidden_dim_lm', type=int, default=model_params['hdl'], help='Hidden dims in latent map')

    ## Training
    parser.add_argument('--lr', type=float, default=train_params['lr'], help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=train_params['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=train_params['epochs'], help='Number of training epochs')
    parser.add_argument('--batch_norm', type=int, default=train_params['batch_norm'], help='Whether to use batch normalisation')

    ## Loss
    parser.add_argument('--lam', type=float, default=train_params['lam'], help="Weight between energy loss in Y and latent space")
    parser.add_argument('--alpha', type=float, default=train_params['alpha'], help="Weight between AE reconstruction energy loss and energy loss of d(lm(X)) in Y ")
    parser.add_argument('--include_pen_e', type=int, default=train_params['include_pen_e'], help='Whether to include KL-like (penalty_e) loss')

    # Parse arguments
    args = parser.parse_args()
    #settings_file_path = args.settings_file
    print("settings_file:", args.settings_file)
    print("args:", args)

    #encoder = args.encoder
    #in_dim = args.in_dim
    #latent_dim = args.latent_dim
    #num_layers = args.num_layer
    #hidden_dim = args.hidden_dim
    #noise_dim_dec = args.noise_dim_dec
    out_act = None 
    #resblock = bool(args.resblock)

    #in_dim_lm = args.in_dim_lm
    #noise_dim_lm = args.noise_dim_lm
    #num_layers_lm = args.num_layer_lm
    #hidden_dim_lm = args.hidden_dim_lm
    beta=1

    # train settings
    #bn = bool(args.batch_norm)
    #print("bn:", bn)
    #batch_size=args.batch_size
    #include_KL = args.include_KL
    #lam = args.lam
    #lr = args.lr
    #training_epochs=args.epochs

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


    ###############
    ### Logging ###
    ###############
    
    # create save directory
    # define path
    save_dir = f"{settings['paths']['output_dir']}/_device{device}{args.latent_dim}_{args.num_layer}_{args.hidden_dim}_{args.noise_dim_dec}_{args.in_dim_lm}_{args.noise_dim_lm}_{args.num_layer_lm}_{args.hidden_dim_lm}_encoderis{args.encoder}_lambda{args.lam}_alpha{args.alpha}_bs{args.batch_size}_bnis{bool(args.batch_norm)}_lr{args.lr}_pene{args.include_pen_e}/"
    
    # create directory
    os.makedirs(save_dir, exist_ok=True)
    print("save_directory:", save_dir)

    

    # save this script for logging info
    # Path to this script
    current_script = os.path.realpath(__file__)
    
    # Copy the script
    shutil.copy(current_script, f"{save_dir}used_training_script.py")

    # Save settings to a new file for logging
    with open(f"{save_dir}actually_used_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

    # write all args to a json file (for logs)
    log_file_settings_name = os.path.join(save_dir, 'model_and_train_settings.json')
    with open(log_file_settings_name, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    

    

    #################
    ### Load Data ###
    #################
    
    
    ### Load temperature data ###
    #############################
    
    ds_path = os.path.join(settings['paths']['data'], settings['paths']['dataset_trefht'])
    
    ds = xr.open_dataset(ds_path) #settings['dataset_trefht'])

    # set train/validation split 90/10 LE members
    ds_train = ds.isel(time=slice(0, 4769 * 90)) 
    ds_test = ds.isel(time=slice(4769 * 90, 476900))
    
    # transform to torch tensors
    x_tr = ut.data_to_torch(ds_train, "TREFHT")
    x_te = ut.data_to_torch(ds_test, "TREFHT")

    # remove NaNs from Temperature data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)
    
    
    ### Load Z500 ###
    #################
    
    z500_path = os.path.join(settings['paths']['data'], settings['paths']['dataset_z500'])
    
    z500_le = xr.open_dataset(z500_path) #(settings["dataset_z500"])
    predictors_combined_le = z500_le.pseudo_pcs.values


    # train validation split 90/10 LE members
    
    ## train
    train_predictors, mean_train, std_train = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
    X_torch = torch.from_numpy(train_predictors)
    

    ## validation
    validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :], mean_train, std_train)
    X_val_torch = torch.from_numpy(validation_predictors)
    print("validation set shape:", predictors_combined_le[90*4769:, :].shape, mean_train.shape, std_train.shape)




    # create train loader
    train_dataset = TensorDataset(X_torch, x_tr_reduced)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Number of batches: {len(train_loader)}")
    
    # create validation loader
    test_dataset = TensorDataset(X_val_torch, x_te_reduced) # test is actually validation
    test_loader_in = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Number of batches: {len(test_loader_in)}")

    ###################
    ### Build Model ###
    ###################
    
    # Encoder
    if args.encoder == "PCA":
        pca, pcs = pcae.get_PC(ds_train)
        eofs_torch_first = pca.components_[:args.latent_dim, :] # take only the first latent_dim PCs, pca.components_ has shape: (n_components_total, n_features) = (modes, space)
        
        # "encode" by projecting onto PCs
        def model_enc(x, eofs=eofs_torch_first):
            teofs = ((torch.from_numpy(eofs)).T).to(device)
            return torch.matmul(x, teofs)
            
        
    elif args.encoder == "learnable":
        model_enc = StoNet(in_dim=args.in_dim,
                           out_dim=args.latent_dim,
                           num_layer=args.num_layer,
                           hidden_dim=args.hidden_dim,
                           noise_dim=0,
                           add_bn=bool(args.batch_norm),
                           out_act=out_act,
                           resblock=bool(args.resblock)).to(device)
    # Decoder
    model_dec = StoNet(in_dim=args.latent_dim,
                       out_dim=args.in_dim,
                       num_layer=args.num_layer,
                       hidden_dim=args.hidden_dim,
                       noise_dim=args.noise_dim_dec,
                       add_bn=bool(args.batch_norm),
                       out_act=out_act,
                       resblock=bool(args.resblock)).to(device)
    # Latent Map
    model_pred = StoNet(in_dim=args.in_dim_lm,
                        out_dim=args.latent_dim, 
                        num_layer=args.num_layer_lm,
                        hidden_dim=args.hidden_dim_lm, 
                        noise_dim=args.noise_dim_lm,
                        add_bn=bool(args.batch_norm), 
                        out_act=out_act, 
                        resblock=bool(args.resblock)).to(device)
    
    # print model state dict
    print("MODEL INITIALISATION")
    #for name, tensor in model_enc.state_dict().items():
    #    print(f"{name}: {tensor.shape}")
    
    # OPTIMIZER
    if args.encoder == "learnable":
        optimizer_enc = torch.optim.Adam(model_enc.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(model_dec.parameters(), lr=args.lr)
    optimizer_pred = torch.optim.Adam(model_pred.parameters(), lr=args.lr)

    

    

    ################
    ### Training ###
    ################
        
    # log file
    log_file_name = os.path.join(save_dir, 'log.txt')
    log_file = open(log_file_name, "wt")
    log = (
                f"Epoch, "
                f"Total loss, "
                f"Total S1, "
                f"Total S2, "
                f"DPA NRGY, "
                f"DPA s1, "
                f"DPA s2, "
                f"LM NRGY, "
                f"LM s1, "
                f"LM s2 ")
    
    # write log file 
    log_file.write(log + '\n')
    log_file.flush()
    
    # Loss lists
    epochs_list=[]
    
    ## total losses
    loss_total_te=[]
    loss_total_tr=[]
    
    loss_s1_te_total=[]
    loss_s2_te_total=[]
    loss_s1_tr_total = []
    loss_s2_tr_total = []
    
    ## latent map losses
    loss_lin_lat_pred_te=[]
    loss_lin_lat_pred_tr = []
    
    loss_s1_lin_lat_pred_tr = []
    loss_s2_lin_lat_pred_tr = []
    
    loss_s1_lin_lat_pred_te = []
    loss_s2_lin_lat_pred_te = []
    
    
    ## DPA losses
    loss_dpa_te=[]
    loss_dpa_tr=[]
    
    loss_s1_dpa_tr = []
    loss_s2_dpa_tr = []
    
    loss_s1_dpa_te = []
    loss_s2_dpa_te = []
    
    for epoch_idx in range(0, args.epochs):
        print("epoch:", epoch_idx)
        # total losses
        loss_tr = 0
        loss_te = 0
        s1_tr = 0
        s2_tr = 0
        s1_t2 = 0
        s2_t2 = 0
    
        # latent map
        loss_tr_pred = 0
        s1_tr_pred = 0
        s2_tr_pred = 0
        loss_te_pred = 0
        s1_te_pred = 0
        s2_te_pred = 0
    
        # DPA
        loss_tr_dpa = 0
        s1_tr_dpa = 0
        s2_tr_dpa = 0
        loss_te_dpa = 0
        s1_te_dpa = 0
        s2_te_dpa = 0

        if args.encoder == "learnable":
            model_enc.train()
        model_dec.train()
        model_pred.train()
        
        for batch_idx, train_batch in enumerate(train_loader):
            optimizer_dec.zero_grad()
            optimizer_pred.zero_grad()
            if args.encoder == "learnable":
                optimizer_enc.zero_grad()
    
            x = train_batch[0].to(device)
            y = train_batch[1].to(device)
            
            e = model_enc(y)
            rec1 = model_dec(e) 
            rec2 = model_dec(e)
            
            z1 = model_pred(x)
            gen1 = model_dec(z1)
            z2 = model_pred(x)
            gen2 = model_dec(z2)
    
            # FL
            # for plotting
            #z3 = model_pred(x)
            #gen3 = model_dec(z3)
            
            loss_rec, s1_rec, s2_rec = energy_loss_two_sample(y, rec1, rec2, verbose=True, beta=beta)
            loss = loss_rec
            
            #if include_KL:
            # energy loss of predicted z
            loss_pred_z, s1_pred_z, s2_pred_z = energy_loss_two_sample(e, z1, z2, verbose=True, beta=beta) #loss im latent space
            loss_pre = loss_rec + args.lam * loss_pred_z
            #penalty_gen = torch.linalg.vector_norm(z1.std(dim=0) - 1) + torch.linalg.vector_norm(z1.mean(dim=0)) # penalizes deviations from std=1 and mean = 0 for predicted latent sample

            if args.encoder == "learnable":
                if args.include_pen_e:
                    print("penalty e included")
                    penalty_e = torch.linalg.vector_norm(e.std(dim=0) - 1) + torch.linalg.vector_norm(e.mean(dim=0)) # penalizes deviations from std=1 and mean = 0 for encoded sample
                    loss = loss_pre + penalty_e #+ penalty_gen

                else:
                    loss = loss_pre #+ penalty_gen
                
            else:
                loss = loss_pre
            #else:
            loss_pred, s1_pred, s2_pred = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=beta)
            
            # welchen der beiden folgenden losses nutzen?
            # loss = loss_pred 
            loss = loss + args.alpha * loss_pred
                
            loss.backward()
            
            #optimizer.step() #single optimizer
            optimizer_dec.step()
            optimizer_pred.step()
            if args.encoder == "learnable":
                optimizer_enc.step()
            
            
    
            # loss total
            loss_tr += loss.item()
            s1_tr += s1_rec.item() + s1_pred.item()
            s2_tr += s2_rec.item() + s2_pred.item()
    
            # loss DPA
            loss_tr_dpa += loss_rec.item()
            s1_tr_dpa += s1_rec.item()
            s2_tr_dpa += s2_rec.item()
    
            # loss latent map
            loss_tr_pred += loss_pred.item()
            s1_tr_pred += s1_pred.item()            
            s2_tr_pred += s2_pred.item()
                
        # compute test losses every epoch 
        log = (
            f"Train {epoch_idx + 1}, "
            f"{loss_tr / len(train_loader):.4f}, "
            f"{s1_tr / len(train_loader):.4f}, "
            f"{s2_tr  / len(train_loader):.4f}, "
            f"{loss_tr_dpa / len(train_loader):.4f}, "
            f"{s1_tr_dpa / len(train_loader):.4f}, "
            f"{s2_tr_dpa / len(train_loader):.4f}, "
            f"{loss_tr_pred / len(train_loader):.4f}, "
            f"{s1_tr_pred / len(train_loader):.4f}, "
            f"{s2_tr_pred / len(train_loader):.4f}\n")

        
        if args.encoder == "learnable":
            model_enc.eval()
        model_dec.eval()
        model_pred.eval()
        loss_te = 0; loss_pred_te = 0; s1_te = 0; s2_te = 0
        with torch.no_grad():
            for batch_idx_test, test_batch in enumerate(test_loader_in):
                
                x_te = test_batch[0].to(device)
                y_te = test_batch[1].to(device)

                
                # "save latent space"
                encoded_sample = model_enc(y_te)
                #torch.save(encoded_sample, f"{save_dir}epoch_{epoch_idx}_test_sample_latent_space.pt")
                
                rec1_te = model_dec(encoded_sample)
                rec2_te = model_dec(encoded_sample)
                
                gen1_te = model_dec(model_pred(x_te))
                gen2_te = model_dec(model_pred(x_te))
                #gen3_te = model_dec(model_pred(x_te))

                loss_te_rec_pre = energy_loss_two_sample(y_te, rec1_te, rec2_te, beta=beta)
                loss_te_pred_pre = energy_loss_two_sample(y_te, gen1_te, gen2_te, beta=beta)

                # total losses
                loss_te += loss_te_rec_pre[0].item()
                loss_te += loss_te_pred_pre[0].item()
                
                s1_te += loss_te_rec_pre[1].item() + loss_te_pred_pre[1].item() 
                s2_te += loss_te_rec_pre[2].item() + loss_te_pred_pre[2].item()

                # latent map
                loss_te_pred += loss_te_pred_pre[0].item()
                s1_te_pred += loss_te_pred_pre[1].item()
                s2_te_pred += loss_te_pred_pre[2].item()

                # DPA
                loss_te_dpa += loss_te_rec_pre[0].item()
                s1_te_dpa += loss_te_rec_pre[1].item()
                s2_te_dpa += loss_te_rec_pre[2].item()
                
        log += (
            f"Test {epoch_idx + 1}, "
            f"{loss_te / len(test_loader_in):.4f}, "
            f"{s1_te / len(test_loader_in):.4f}, "
            f"{s2_te  / len(test_loader_in):.4f}, "
            f"{loss_te_dpa / len(test_loader_in):.4f}, "
            f"{s1_te_dpa / len(test_loader_in):.4f}, "
            f"{s2_te_dpa / len(test_loader_in):.4f}, "
            f"{loss_te_pred / len(test_loader_in):.4f}, "
            f"{s1_te_pred / len(test_loader_in):.4f}, "
            f"{s2_te_pred / len(test_loader_in):.4f} ")
        if args.encoder == "learnable":
            model_enc.train()
        model_dec.train()
        model_pred.train()
            
        log_file.write(log + '\n')
        log_file.flush()
        
            
        ############################
        ### Append to Loss Lists ###
        ############################
        epochs_list.append(epoch_idx)
    
        # total losses
        loss_total_tr.append(loss_tr / len(train_loader))
        loss_total_te.append(loss_te / len(test_loader_in))
    
        loss_s1_tr_total.append(s1_tr / len(train_loader))
        loss_s2_tr_total.append(s2_tr / len(train_loader))
            
        loss_s1_te_total.append(s1_te / len(test_loader_in))
        loss_s2_te_total.append(s2_te / len(test_loader_in))
    
        # latent map losses
        loss_lin_lat_pred_tr.append(loss_tr_pred / len(train_loader))
        loss_lin_lat_pred_te.append(loss_te_pred / len(test_loader_in))
    
        loss_s1_lin_lat_pred_tr.append(s1_tr_pred  / len(train_loader))
        loss_s2_lin_lat_pred_tr.append(s2_tr_pred / len(train_loader))
        
        loss_s1_lin_lat_pred_te.append(s1_te_pred/len(test_loader_in))
        loss_s2_lin_lat_pred_te.append(s2_te_pred/len(test_loader_in))
    
        ## DPA losses
        loss_dpa_te.append(loss_te_dpa / len(test_loader_in))
        loss_dpa_tr.append(loss_tr_dpa / len(train_loader))
        
        loss_s1_dpa_tr.append(s1_tr_dpa / len(train_loader))
        loss_s2_dpa_tr.append(s2_tr_dpa /len(train_loader))
        
        loss_s1_dpa_te.append(s1_te_dpa/len(test_loader_in))
        loss_s2_dpa_te.append(s2_te_dpa/len(test_loader_in))
        
        ##################
        ### Save Model ###
        ##################
        if (epoch_idx + 1) % 5 == 0:
            if args.encoder == "learnable":
                torch.save(model_enc.state_dict(), save_dir + "model_enc_" + str(epoch_idx + 1) + ".pt")
            torch.save(model_dec.state_dict(), save_dir + "model_dec_" + str(epoch_idx + 1) + ".pt")
            torch.save(model_pred.state_dict(), save_dir + "model_pred_" + str(epoch_idx + 1) + ".pt")

        
        ############
        ### Plot ###
        ############
        # save loss curves in last epoch
        if epoch_idx == int(args.epochs-1):
            ut.plot_all_losses(
                            loss_total_tr, loss_total_te, loss_s1_tr_total, loss_s2_tr_total,
                            loss_s1_te_total, loss_s2_te_total,
                            loss_dpa_tr, loss_dpa_te, loss_s1_dpa_tr, loss_s2_dpa_tr,
                            loss_s1_dpa_te, loss_s2_dpa_te,
                            loss_lin_lat_pred_tr, loss_lin_lat_pred_te,
                            loss_s1_lin_lat_pred_tr, loss_s2_lin_lat_pred_tr,
                            loss_s1_lin_lat_pred_te, loss_s2_lin_lat_pred_te,
                            args.epochs)
            plt.savefig(f"{save_dir}final_losses.pdf", format='pdf')


if __name__ == "__main__":
    main()
