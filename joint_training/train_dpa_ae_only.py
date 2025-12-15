import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
#from torchvision.utils import make_grid
from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss_two_sample
import argparse
import json
import xarray as xr
import torch.nn as nn
#from sklearn.manifold import TSNE
#import pca_encoder as pcae


import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil
import sys
sys.path.append('../')
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
    parser.add_argument('--resblock', type=bool, default=True, help="Whether to use residual block")
    
    parser.add_argument('--in_dim_lm', type=int, default=1001, help="Input dimension of the latent map")
    parser.add_argument('--noise_dim_lm', type=int, help='Dimension of noise added in latent map')
    parser.add_argument('--num_layer_lm', type=int, help='Number of layers in latent map')
    parser.add_argument('--hidden_dim_lm', type=int, help='Hidden dims in latent map')

    ## Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_norm', type=int, default=0, help='Whether to use batch normalisation.')

    ## Loss
    parser.add_argument('--lam', type=float, default=1.0, help="Weight between energy loss in Y and latent space")
    #parser.add_argument('--include_KL', type=bool, default="False", help='Whether to include KL loss')

    # Parse arguments
    args = parser.parse_args()
    settings_file_path = args.settings_file
    print("settings_file:", settings_file_path)

    encoder = args.encoder
    in_dim = args.in_dim
    latent_dim = args.latent_dim
    num_layers = args.num_layer
    hidden_dim = args.hidden_dim
    noise_dim_dec = args.noise_dim_dec
    out_act = None#args.out_activation
    resblock = args.resblock

    in_dim_lm = args.in_dim_lm
    noise_dim_lm = args.noise_dim_lm
    num_layers_lm = args.num_layer_lm
    hidden_dim_lm = args.hidden_dim_lm
    beta=1

    # train settings
    bn = bool(args.batch_norm)
    batch_size=args.batch_size
    #include_KL = args.include_KL
    lam = args.lam
    lr = args.lr
    training_epochs=args.epochs


    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    
    # create save directory
    save_dir = f"{settings['output_dir']}autoencode_only_{latent_dim}_{num_layers}_{hidden_dim}_{noise_dim_dec}_{in_dim_lm}_{noise_dim_lm}_{num_layers_lm}_{hidden_dim_lm}_encoderis{encoder}_lambda{lam}_bs{batch_size}_bnis{bn}/"
    
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

    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
      
    # Save to a new file for logging
    with open(f"used_settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    
    # Load temperature data
    ds = xr.open_dataset(settings['dataset_trefht'])
    print("Dataset:", settings['dataset_trefht'])

    # set train/test split
    ds_train = ds.isel(time=slice(0, 4769 * 10)) #4769 * 80
    ds_test = ds.isel(time=slice(4769 * 10, 476900)) #4769 * 80

    print(ds_train.TREFHT.shape)
    print(ds_test.TREFHT.shape)
    
    
    # transform to torch tensors
    x_tr = ut.data_to_torch(ds_train, "TREFHT")
    x_te = ut.data_to_torch(ds_test, "TREFHT")

    print(x_tr.shape)
    print(x_te.shape)
    
    # load Z500
    #ds_z500_pre = xr.open_dataset(settings['dataset_z500'])
    #ds_z500, _, _ = ut.standardize_numpy(ds_z500_pre.pseudo_pcs.values)
    #print("z500 shape", ds_z500.shape)
    #z500 = torch.from_numpy(ds_z500)
    #print("z500 shape", z500.shape)
    
    
    #z500_train = z500[:int(128000),:]
    #z500_test = z500[int(-64000):,:]

    # remove NaNs from data
    x_tr_reduced, mask_x_tr = ut.remove_nan_columns(x_tr)
    x_te_reduced, mask_x_te = ut.remove_nan_columns(x_te)

    # create data loader Temperature
    train_dataset = TensorDataset(x_tr_reduced)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of batches: {len(train_loader)}")
    
    # create test loader Temperature
    test_dataset = TensorDataset(x_te_reduced)
    test_loader_in = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of batches: {len(test_loader_in)}")

    ###################
    ### Build Model ###
    ###################
    # model parameters 
    
    # Encoder
    if encoder == "PCA":
        pca, pcs = pcae.get_PC(ds_train)
        eofs_torch_first = pca.components_[:latent_dim, :] # # take only the first latent_dim PCs, pca.components_ has shape: (n_components_total, n_features) = (modes, space)
        
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
    
    # print model state dict
    print("MODEL INITIALISATION")
    for name, tensor in model_enc.state_dict().items():
        print(f"{name}: {tensor.shape}")
    
    # set one combined or individual optimizers
    #optimizer = torch.optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_pred.parameters()), lr=lr)
    if args.encoder == "learnable":
        optimizer_enc = torch.optim.Adam(model_enc.parameters(), lr=args.lr)
    optimizer_dec = torch.optim.Adam(model_dec.parameters(), lr=args.lr)
    optimizer_pred = torch.optim.Adam(model_pred.parameters(), lr=args.lr)

    

    

    ################
    ### Training ###
    ################
    
    # training settings
    print_every_nepoch = 1
    plot_every_epoch = 5    
    
    # log file
    log_file_name = os.path.join(save_dir, 'log.txt')
    log_file = open(log_file_name, "wt")
    log = (
                f"Epoch, "
                f"Total loss, "
                f"Total S1, "
                f"Total S2 ")
    
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
    
    for epoch_idx in range(0, training_epochs):
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

        if encoder == "learnable":
            model_enc.train()
        model_dec.train()
        model_pred.train()
        
        for batch_idx, train_batch in enumerate(train_loader):
            #optimizer.zero_grad()
            optimizer_dec.zero_grad()
            optimizer_pred.zero_grad()
            if args.encoder == "learnable":
                optimizer_enc.zero_grad()
    
            #x = train_batch[0].to(device)
            y = train_batch[0].to(device)
            
            e = model_enc(y)
            rec1 = model_dec(e) 
            rec2 = model_dec(e)
            
            #z1 = model_pred(x)
            #gen1 = model_dec(z1)
            #z2 = model_pred(x)
            #gen2 = model_dec(z2)
    
            # FL
            # for plotting
            #z3 = model_pred(x)
            #gen3 = model_dec(z3)
            
            loss_rec, s1_rec, s2_rec = energy_loss_two_sample(y, rec1, rec2, verbose=True, beta=beta)
            #loss = loss_rec
            
            #if include_KL:
            # energy loss of predicted z
            #loss_pred_z, s1_pred_z, s2_pred_z = energy_loss_two_sample(e, z1, z2, verbose=True, beta=beta) #loss im latent space
            #loss_pre = loss_rec + lam * loss_pred_z
            #penalty_gen = torch.linalg.vector_norm(z1.std(dim=0) - 1) + torch.linalg.vector_norm(z1.mean(dim=0)) # penalizes deviations from std=1 and mean = 0 for predicted latent sample

            if encoder == "learnable":
                penalty_e = torch.linalg.vector_norm(e.std(dim=0) - 1) + torch.linalg.vector_norm(e.mean(dim=0)) # penalizes deviations from std=1 and mean = 0 for encoded sample
                #loss = loss_pre + penalty_e #+ penalty_gen

            else:
                loss = loss_pre #+ penalty_gen
                
            
            #else:
            #loss_pred, s1_pred, s2_pred = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=beta)
            
            # welchen der beiden folgenden losses nutzen?
            # loss = loss_pred 
            loss = loss_rec + penalty_e
                
            loss.backward()
            
            #optimizer.step() #single optimizer
            optimizer_dec.step()
            optimizer_pred.step()
            if encoder == "learnable":
                optimizer_enc.step()
            
            
    
            # loss total
            loss_tr += loss.item()
            s1_tr += s1_rec.item() #+ s1_pred.item()
            s2_tr += s2_rec.item() #+ s2_pred.item()
    
            # loss DPA
            #loss_tr_dpa += loss_rec.item()
            #s1_tr_dpa += s1_rec.item()
            #s2_tr_dpa += s2_rec.item()
    
            # loss latent map
            #loss_tr_pred += loss_pred.item()
            #s1_tr_pred += s1_pred.item()            
            #s2_tr_pred += s2_pred.item()
        
        #if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0):
        
        # compute test losses every epoch 
        log = (
            f"Train {epoch_idx + 1}, "
            f"{loss_tr / len(train_loader):.4f}, "
            f"{s1_tr / len(train_loader):.4f}, "
            f"{s2_tr  / len(train_loader):.4f}\n")

        
        if encoder == "learnable":
            model_enc.eval()
        model_dec.eval()
        model_pred.eval()
        loss_te = 0; loss_pred_te = 0; s1_te = 0; s2_te = 0
        with torch.no_grad():
            for batch_idx_test, test_batch in enumerate(test_loader_in):
                
                #x_te = test_batch[0].to(device)
                y_te = test_batch[0].to(device)

                
                # "save latent space"
                encoded_sample = model_enc(y_te)
                torch.save(encoded_sample, f"{save_dir}epoch_{epoch_idx}_test_sample_latent_space.pt")
                
                rec1_te = model_dec(encoded_sample)
                rec2_te = model_dec(encoded_sample)
                
                #gen1_te = model_dec(model_pred(x_te))
                #gen2_te = model_dec(model_pred(x_te))
                #gen3_te = model_dec(model_pred(x_te))

                loss_te_rec_pre = energy_loss_two_sample(y_te, rec1_te, rec2_te, beta=beta)
                #loss_te_pred_pre = energy_loss_two_sample(y_te, gen1_te, gen2_te, beta=beta)

                # total losses
                loss_te += loss_te_rec_pre[0].item()
                #loss_te += loss_te_pred_pre[0].item()
                
                s1_te += loss_te_rec_pre[1].item() #+ loss_te_pred_pre[1].item() 
                s2_te += loss_te_rec_pre[2].item() #+ loss_te_pred_pre[2].item()

                # latent map
                #loss_te_pred += loss_te_pred_pre[0].item()
                #s1_te_pred += loss_te_pred_pre[1].item()
                #s2_te_pred += loss_te_pred_pre[2].item()

                # DPA
                #loss_te_dpa += loss_te_rec_pre[0].item()
                #s1_te_dpa += loss_te_rec_pre[1].item()
                #s2_te_dpa += loss_te_rec_pre[2].item()
                
        log += (
            f"Test {epoch_idx + 1}, "
            f"{loss_te / len(test_loader_in):.4f}, "
            f"{s1_te / len(test_loader_in):.4f}, "
            f"{s2_te  / len(test_loader_in):.4f}")
            
        if encoder == "learnable":
            model_enc.train()
        model_dec.train()
        model_pred.train()
            
        log_file.write(log + '\n')
        log_file.flush()
        
        ############
        ### PLOT ###
        ############
        if (epoch_idx == 0 or (epoch_idx + 1) % plot_every_epoch == 0):
            #print(batch_idx_test)
            if batch_idx_test == 499:
                print("y_te shape",y_te.shape)
                y_te_from_restored = ut.restore_nan_columns(y_te, mask_x_te)
                print("y_te_from_restored shape:", y_te_from_restored.shape)
                gen1_te_from_restored = ut.restore_nan_columns(rec1_te, mask_x_te)
                gen2_te_from_restored = ut.restore_nan_columns(rec2_te, mask_x_te)
                print("gen1_te_from_restored:", gen1_te_from_restored.shape)

                #gen3_te_from_restored = ut.restore_nan_columns(gen3_te, mask_x_te)
        
        
                
                timesteps=[15, 42, 60, 94, 110]
                y_te_da = ut.torch_to_dataarray_ae_only(y_te_from_restored, coords_ds=ds_test, name="Temperature")
                reconstructed_da_11 = ut.torch_to_dataarray_ae_only(gen1_te_from_restored, coords_ds=ds_test, name="Temperature")
                reconstructed_da_12 = ut.torch_to_dataarray_ae_only(gen2_te_from_restored, coords_ds=ds_test, name="Temperature")
                #reconstructed_da_21 = ut.torch_to_dataarray_ae_only(gen3_te_from_restored, coords_ds=ds_test, name="Temperature")
                
                fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})
                axs = axs.flatten()  # Flatten for easy indexing
                #print(axs)
                list_of_dataarrays_re = [reconstructed_da_11, reconstructed_da_12]
                for i, (ax, t) in enumerate(zip(axs, timesteps)):
                    if i ==0:
                        ut.plot_temperature_panel(axs[0], y_te_da.isel(time=t), vmax_shared=5, sample_nr = "Truth")
                        ut.plot_temperature_panel(axs[5], list_of_dataarrays_re[0].isel(time=t), vmax_shared=5, sample_nr = "Sample 1")
                        ut.plot_temperature_panel(axs[10], list_of_dataarrays_re[1].isel(time=t), vmax_shared=5, sample_nr = "Sample 2")
                        #ut.plot_temperature_panel(axs[15], list_of_dataarrays_re[2].isel(time=t), vmax_shared=5, sample_nr = "Sample 3")
                        
                    else:
                        ut.plot_temperature_panel(ax, y_te_da.isel(time=t), vmax_shared=5)
                        ut.plot_temperature_panel(axs[i+5], list_of_dataarrays_re[0].isel(time=t), vmax_shared=5)
                        ut.plot_temperature_panel(axs[i+10], list_of_dataarrays_re[1].isel(time=t), vmax_shared=5)
                        #ut.plot_temperature_panel(axs[i+15], list_of_dataarrays_re[2].isel(time=t), vmax_shared=5)
                        
                
                # Optional: Add a colorbar
                cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='horizontal', fraction=0.03, pad=0.05)
                cbar.set_label('Temperature (K)')
                
                
                plt.savefig(f"{save_dir}epoch_{epoch_idx}_latent_map_decoded_samples.pdf", format='pdf')
                plt.show()
            
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
        loss_lin_lat_pred_tr.append(0)
        loss_lin_lat_pred_te.append(0)
    
        loss_s1_lin_lat_pred_tr.append(0)
        loss_s2_lin_lat_pred_tr.append(0)
        
        loss_s1_lin_lat_pred_te.append(0)
        loss_s2_lin_lat_pred_te.append(0)
    
        ## DPA losses
        loss_dpa_te.append(0)
        loss_dpa_tr.append(0)
        
        loss_s1_dpa_tr.append(0)
        loss_s2_dpa_tr.append(0)
        
        loss_s1_dpa_te.append(0)
        loss_s2_dpa_te.append(0)
        
        ##################
        ### Save Model ###
        ##################
        if (epoch_idx + 1) % 10 == 0:
            torch.save(model_enc.state_dict(), save_dir + "model_enc_" + str(epoch_idx + 1) + ".pt")
            torch.save(model_dec.state_dict(), save_dir + "model_dec_" + str(epoch_idx + 1) + ".pt")
            torch.save(model_pred.state_dict(), save_dir + "model_pred_" + str(epoch_idx + 1) + ".pt")

        
        ############
        ### Plot ###
        ############
        # save loss curves in last epoch
        if epoch_idx == int(training_epochs-1):
            ut.plot_all_losses(
                            loss_total_tr, loss_total_te, loss_s1_tr_total, loss_s2_tr_total,
                            loss_s1_te_total, loss_s2_te_total,
                            loss_dpa_tr, loss_dpa_te, loss_s1_dpa_tr, loss_s2_dpa_tr,
                            loss_s1_dpa_te, loss_s2_dpa_te,
                            loss_lin_lat_pred_tr, loss_lin_lat_pred_te,
                            loss_s1_lin_lat_pred_tr, loss_s2_lin_lat_pred_tr,
                            loss_s1_lin_lat_pred_te, loss_s2_lin_lat_pred_te,
                            training_epochs)
            plt.savefig(f"{save_dir}final_losses.pdf", format='pdf')


if __name__ == "__main__":
    main()
