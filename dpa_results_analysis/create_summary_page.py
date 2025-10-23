from reportlab.lib.pagesizes import A2, A3, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import glob
import argparse

def main(args=None):
    
    parser = get_parser()

    # if called as standalone script
    if args is None:
        args = parser.parse_args()

    summary(path_eth=args.save_path_eth,
           path_le=args.save_path_le,
           save_path=args.save_path,
           period=args.period,
           comment=args.summary_save_comment)

def get_parser():
    """Return an argument parser for this module."""
    
    parser = argparse.ArgumentParser(description="Analyse DPA ensemble from LE train set")
    parser.add_argument("--save_path_eth", type=str, required=True, help="Save path for ETH analysis figures")
    parser.add_argument("--save_path_le", type=str, required=True, help="Save path for ETH analysis figures")
    parser.add_argument("--save_path", type=str, help = "Where to save summary page.")
    parser.add_argument("--period", type=str, help = "Period")
    parser.add_argument("--summary_save_comment", type=str, default="", help="Comment to include in saved summary file name.")



    return parser

def summary(path_eth, path_le, save_path, period, include_train_analysis=1, comment=""):
    images_new_pre = [f"{path_le}/LE_train_set_energy_loss_map.png", # 0
              f"{path_le}/LE_train_set_S1_loss_map.png", # 1
              f"{path_le}/LE_train_set_S2_loss_map.png", # 2
              f"{path_le}/all_losses.png", # 3
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
              f"{path_le}/eloss_and_aerosols.png", # 19
              
              
             ]

    print("Images:", images_new_pre)
        
    if not include_train_analysis:
        images_new = [x for x in images_new_pre if "LE" not in x]
    else:
        images_new = images_new_pre

    print("Images:", images_new)
    
    c = canvas.Canvas(f"{save_path}/{period}_evaluation_combined_{comment}.pdf", pagesize=landscape(A2))
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

if __name__ == "__main__":
    main()