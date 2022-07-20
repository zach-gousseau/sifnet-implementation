import sys
import os
import datetime
import xarray as xr
import numpy as np
import logging

sys.path.append(os.path.join(sys.path[0], '..'))

from data_generator import DataGen
from train_pinn import read_and_combine_glorys_era5, X_VARS

month=5
num_timesteps_input=3
num_timesteps_predict=4
num_training_years=1

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    ds = read_and_combine_glorys_era5(
        era5_path='/home/zgoussea/scratch/era5_hb_daily.zarr',
        glorys_path='/home/zgoussea/scratch/glorys12/glorys12_v2.zarr',
        start_year=1993,
        end_year=1996,
        lat_range=(51, 70),  # Hudson Bay
        lon_range=(-95, -65),  # Hudson Bay
        coarsen=4,
    )
    
    data_gen = DataGen()

    # Read data
    ds = data_gen.get_data(
        ds=ds,
        add_add=False,
        X_vars=X_VARS,
        Y_vars=['sivol']
        )
    
    # Get landmask from SIC
    landmask = data_gen.get_landmask_from_nans(ds, var_='sivol')

    preds = None
    loss_curves = None

    # Create dataset iterator
    datasets = data_gen.get_generator(
        ds,
        month=5,
        num_timesteps_input=2,
        num_timesteps_predict=3,
        binary_sic=False,
        num_training_years=1,
    )
    
    for data in datasets:
        with open('debug.npy', 'wb') as f:
            np.save(f, data['test_X'])
            np.save(f, data['test_Y'])
        
        break