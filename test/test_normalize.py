import sys
import os
import datetime
from sklearn.preprocessing import StandardScaler
import xarray as xr
import numpy as np

sys.path.append(os.path.join(sys.path[0], '..'))

from data_generator import DataGen

month=5
num_timesteps_input=3
num_timesteps_predict=4
num_training_years=1

if __name__ == '__main__':
    # Read data -----------------------
    era5 = xr.open_zarr('/home/zgoussea/scratch/era5_hb_daily.zarr')

    # Smaller area for testing
    era5 = era5.sel(latitude=slice(55, 60), longitude=slice(-90, -85))

    # Smaller temporal window for testing
    s, e = datetime.datetime(1993, 1, 1), datetime.datetime(1996, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    
    data_gen = DataGen()
    
    # Read data
    ds = data_gen.get_data(
        ds=era5,
        add_add=False,
        X_vars=list(era5.data_vars)[:4] + ['siconc'],
        Y_vars=['siconc']
        )
    
    print(11)
    fn = np.datetime_as_string(ds.time.values[0])[:10] + '-' + \
        np.datetime_as_string(ds.time.values[0])[:10] + '-' + \
        ','.join(list(ds.data_vars))
    print(ds.mean())
    ds.std().to_netcdf(f'cache/mean_{fn}.nc')
    ds.mean().to_netcdf(f'cache/std_{fn}.nc')
    ds = (ds - ds.mean()) / ds.std()
    print(12)

    # Create expanded dataset (add timesteps)
    ds_timesteps = ds.rolling(
        time=num_timesteps_input + num_timesteps_predict
    ).construct("timesteps")

    # Remove first num_timesteps_input timesteps and assign the launch date
    # dates to be the de-facto dates for each timestep.
    launch_dates = ds_timesteps.time[num_timesteps_input:-num_timesteps_predict]
    ds_timesteps = ds_timesteps.isel(
        time=slice(num_timesteps_input + num_timesteps_predict, None)
    )
    ds_timesteps = ds_timesteps.assign_coords(time=launch_dates)

    ds_train, ds_test, ds_valid = next(data_gen.data_split(ds_timesteps, num_training_years, month))
    

    # Update the normalization scaler with just the first timestep of the train array
    # data_gen.update_scaler(np.array(ds_train.isel(timesteps=0).to_array()))

    # Convert to numpy & replace NaNs with 0s
    # valid_array = np.nan_to_num(np.array(ds_train.to_array()))
    valid_array = np.array(ds_valid.to_array())

    valid_X, valid_Y = data_gen.split_xy(valid_array, num_timesteps_predict, num_timesteps_input, split_index=len(data_gen.Y_vars))
    
    print('Pre-normalize')
    for i in range(valid_X.shape[-1]):
        print(i, np.nanmean(valid_X[..., i]), np.nanstd(valid_X[..., i]))

    # 0
    data_gen.scaler = StandardScaler()
    data_gen.update_scaler(valid_array)
        
    valid_X_0 = data_gen.normalize(valid_X)
    print('Post-normalize (0)')
    for i in range(valid_X.shape[-1]):
        print(i, np.nanmean(valid_X_0[..., i]), np.nanstd(valid_X_0[..., i]))

    # 1
    data_gen.scaler = StandardScaler()
    data_gen.update_scaler(np.array(ds_train.isel(timesteps=0).to_array()))
    
    valid_X_1 = data_gen.normalize(valid_X)
    print('Post-normalize (1)')
    for i in range(valid_X.shape[-1]):
        print(i, np.nanmean(valid_X_1[..., i]), np.nanstd(valid_X_1[..., i]))
    
    # 2
    data_gen.scaler = StandardScaler()
    data_gen.update_scaler(np.array(ds_train.to_array()))
    
    valid_X_1 = data_gen.normalize(valid_X)
    print('Post-normalize (1)')
    for i in range(valid_X.shape[-1]):
        print(i, np.nanmean(valid_X_1[..., i]), np.nanstd(valid_X_1[..., i]))