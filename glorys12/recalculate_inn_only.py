import xarray as xr
import glob
import re
import os
import dask
import numpy as np
import matplotlib.pyplot as plt

def intensification(Vol, nsecs):
    inn = (Vol[1:, :, :] - Vol[:-1, :, :]) / (1 * nsecs)
    return np.pad(inn, ((0, 1), (0,0),(0,0)), 'constant', constant_values=np.nan)

def calculate_inn_forward_diff(ds, landmask, suffix=''):
    siconc = np.nan_to_num(ds.siconc.values)
    sithick = np.nan_to_num(ds.sithick.values)
    
    # Replace 0s with NaNs where it does not intersect the landmask
    landmask = np.broadcast_to(landmask, siconc.shape)
    
    siconc[~landmask] = np.nan
    sithick[~landmask] = np.nan
    vol = siconc * sithick
    inn = intensification(vol, 86400)  # Intensification

    ds['inn' + suffix] = (('time', 'latitude', 'longitude'), inn)
    return ds

if __name__ == '__main__':

    inpath = '/home/zgoussea/projects/def-ka3scott/zgoussea/glorys12_v2_with_fluxes.zarr'

    ds = xr.open_dataset(inpath)
    landmask = ~np.isnan(ds.isel(time=0).zos.values)

    i_0 = 0
    stepsize = 10

    i_1 = -99

    while i_1 < len(ds.time):
        i_1 = min(i_0 + stepsize, len(ds.time))
        print(i_1)

        # try:
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_sub = ds.isel(time=slice(max(i_0, 0), i_1 + 1))  # Pad indexes because first and last will be NaNs (intensification)

            ds_sub = calculate_inn_forward_diff(ds_sub, landmask, suffix='_v3')
            
            ds_sub = ds_sub.isel(time=slice(None, -1))  # Remove first and last
            
            ds_sub = ds_sub[['inn_v3']].drop(['depth'])
            
            for var_ in ds_sub.data_vars:
                try:
                    del ds_sub[var_].encoding['chunks']
                except KeyError:
                    pass
                try:
                    del ds_sub[var_].attrs['grid_mapping']
                except KeyError:
                    pass
                
            ds_sub = ds_sub.chunk({'longitude': -1, 'latitude': -1, 'time': -1})

            outpath = '/home/zgoussea/projects/def-ka3scott/zgoussea/glorys12_v2_with_fluxes_inn_v3.zarr'
            if os.path.exists(outpath):
                ds_sub.to_zarr(outpath, append_dim='time')
            else:
                ds_sub.to_zarr(outpath)

            # coords_region = {
            #     'latitude': slice(0, ds.latitude.size),
            #     'longitude': slice(0, ds.longitude.size),
            #     'time': slice(max(i_0, 0), i_1)
            #     }
            
            # # var_region = {var_: slice(0, ds[var_].size) for var_ in ds}

            # ds_sub.to_zarr(outpath, mode='a', region=coords_region)

        i_0 = i_1