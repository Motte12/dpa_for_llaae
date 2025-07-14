import xarray as xr
import json
import numpy as np
from sklearn.decomposition import PCA

from sklearn.decomposition import PCA
import xarray as xr

def get_PC(ds_train):
    """
    Perform Principal Component Analysis (PCA) on the `TREFHT` variable of an input 
    xarray.Dataset with dimensions (time, lat, lon).

    The function:
      - Stacks the spatial dimensions (lat, lon) into a single "space" dimension,
      - Removes spatial locations (columns) where all time values are NaN,
      - Centers the data by subtracting the temporal mean at each spatial location,
      - Applies PCA across the space dimension, treating time as the sample axis.

    Parameters:
        ds_train (xarray.Dataset): Input dataset containing the variable `TREFHT` 
                                   with dimensions (time, lat, lon).

    Returns:
        pcs (numpy.ndarray): Principal component scores (time series) with shape 
                             (time, n_components), where each column corresponds to 
                             one principal component.

    Notes:
        - The function does not return the PCA model or the EOFs (spatial patterns). 
          These can be added to the return statement if needed for reconstruction 
          or projection of new data.
        - Assumes that `TREFHT` is present in `ds_train`.
    """

    # Stack spatial dimensions into a single 'space' dimension → shape: (time, space)
    ds_stacked = ds_train.stack(space=("lat", "lon"))

    # Select the TREFHT variable and subtract temporal mean to center the data
    stacked_anom = (ds_stacked - ds_stacked.mean(dim="time")).TREFHT
    print(stacked_anom.shape)  # Debug: show shape after stacking

    # Remove spatial locations (columns) that are entirely NaN over time
    da_clean = stacked_anom.sel(space=~stacked_anom.isnull().all(dim='time'))
    print("Shape after removing NANs:", da_clean.shape)
    
    # Perform PCA on the cleaned, centered data
    pca = PCA()
    pcs = pca.fit_transform(da_clean)  # Output shape: (time, n_components)

    return pca, pcs

