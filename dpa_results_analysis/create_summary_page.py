from reportlab.lib.pagesizes import A2, A3, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import glob

def summary(path_eth, save_path, period):
    images_new = [f"ETH_analysis_results/final_analysis_train_LE/LE_train_set_energy_loss_map.png", # 0
              f"ETH_analysis_results/final_analysis_train_LE/LE_train_set_S1_loss_map.png", # 1
              f"ETH_analysis_results/final_analysis_train_LE/LE_train_set_S2_loss_map.png", # 2
              f"ETH_analysis_results/final_analysis_train_LE/all_losses.png", # 3
              f"{path_eth}/LE_Europe_Forced_response.png", # 4
              f"{path_eth}/energy_loss_map.png", # 5
              f"{path_eth}/S1_loss_map.png", # 6
              f"{path_eth}/S2_loss_map.png", # 7
              f"{path_eth}/correlation_eth_test_figure.png", # 8
              f"{path_eth}/energy_loss_time_series_eth_test_set.png", # 9
              f"{path_eth}/cf_energy_loss_map.png", # 10
              f"{path_eth}/cf_S1_loss_map.png", # 11
              f"{path_eth}/cf_S2_loss_map.png", # 12
              f"{path_eth}/cf_correlation_eth_test_figure.png", # 13
              f"{path_eth}/energy_loss_time_series_eth_test_set_cf.png", # 14
              f"{path_eth}/Ger_rank_hist.png", # 15
              f"{path_eth}/Ger_cf_rank_hist.png", # 16
              f"{path_eth}/Sp_rank_hist.png", # 17
              f"{path_eth}/Sp_cf_rank_hist.png", # 18
              f"ETH_analysis_results/final_analysis_train_LE/eloss_and_aerosols.png", # 19
              
              
             ]


    c = canvas.Canvas(f"{save_path}/{period}_evaluation_combined.pdf", pagesize=landscape(A2))
    page_width, page_height = landscape(A2)
    
    ncols, nrows = 5, 4   # portrait fits 2 across, 4 down
    cell_width = page_width / ncols
    cell_height = page_height / nrows
    
    for i, img in enumerate(images_new):
        print(i)
        row = i // ncols
        col = i % ncols
        x = col * cell_width
        y = page_height - (row + 1) * cell_height
        #if i in [13]:
        #    continue
        #else:
        c.drawImage(img, x, y, width=cell_width, height=cell_height,
                        preserveAspectRatio=True, anchor='c')
    
    c.showPage()


    images_p2 = [f"{path_eth}/quantiles/factual_Q-Q_plot_of_dpa_ens_mean_single_grid_cells.png", f"{path_eth}/quantiles/counterfactual_Q-Q_plot_of_dpa_ens_mean_single_grid_cells.png"]
    
    ncols, nrows = 2, 1 
    cell_width = page_width / ncols
    cell_height = page_height / nrows
    
    for i, img in enumerate(images_p2):
        print(i)
        row = i // ncols
        col = i % ncols
        x = col * cell_width
        y = page_height - (row + 1) * cell_height
        #if i in [13]:
        #    continue
        #else:
        c.drawImage(img, x, y, width=cell_width, height=cell_height,
                        preserveAspectRatio=True, anchor='c')

    c.showPage()


    images_p3 = [f"{path_eth}/quantiles/factual_Q-Q_plot_of_dpa_ens_mean_spatial_mean.png", f"{path_eth}/quantiles/factual_Q-Q_plot_mean_of_quantiles_spatial_mean.png", f"{path_eth}/quantiles/counterfactual_Q-Q_plot_of_dpa_ens_mean_spatial_mean.png", f"{path_eth}/quantiles/counterfactual_Q-Q_plot_mean_of_quantiles_spatial_mean.png"]
    
    ncols, nrows = 2, 2 
    cell_width = page_width / ncols
    cell_height = page_height / nrows
    
    for i, img in enumerate(images_p3):
        print(i)
        row = i // ncols
        col = i % ncols
        x = col * cell_width
        y = page_height - (row + 1) * cell_height
        #if i in [13]:
        #    continue
        #else:
        c.drawImage(img, x, y, width=cell_width, height=cell_height,
                        preserveAspectRatio=True, anchor='c')
    
    
    c.save()
