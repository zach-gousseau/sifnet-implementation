import xarray as xr
import glob
import re
import os
import dask
import numpy as np
import matplotlib.pyplot as plt

from calculate_adv import *

if __name__ == '__main__':

    zarrpath = '/home/zgoussea/scratch/glorys12/glorys12_v2.zarr'

    with open('log.log', 'r') as f:
        last_write = int(f.readlines()[-1].strip())

    ds = xr.open_dataset(zarrpath)
    landmask = ~np.isnan(ds.isel(time=0).zos.values)

    i_0 = 0
    stepsize = 10

    i_1 = -99

    while i_1 < len(ds.time):
        i_1 = min(i_0 + stepsize, len(ds.time))
        print(i_1)

        if i_0 <= last_write:
            i_0 = i_1
            continue

        # try:
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_sub = ds.isel(time=slice(max(i_0 - 1, 0), i_1 + 1))  # Pad indexes because first and last will be NaNs (intensification)

            ds_sub = calculate_all(ds_sub, landmask, suffix='_v2')
            print('calculated')
            
            ds_sub = ds_sub.isel(time=slice(1, -1))  # Remove first and last
            # ds_sub = ds_sub.chunk({'longitude': 1080, 'latitude': -1})
            
            ds_sub = ds_sub[['div_v2', 'adv_v2', 'inn_v2', 'res_v2']]#.drop(['depth'])
            
            for var_ in ds_sub.data_vars:
                try:
                    del ds_sub[var_].attrs['grid_mapping']
                except KeyError:
                    pass
                
            outpath = '/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes.zarr'
                
            if os.path.exists(outpath):
                ds_sub.to_zarr(outpath, append_dim='time')
            else:
                ds_sub.to_zarr(outpath)

            # ds_sub.to_zarr(zarrpath,
            #                mode='a',
            #             #    append_dim='time',
            #                region={
            #                    'latitude': slice(0, ds.latitude.size),
            #                    'longitude': slice(0, ds.longitude.size),
            #                    'time': slice(max(i_0 - 1, 0) + 1, i_1 + 1 - 1)
            #                    }
            #                )

        i_0 = i_1

        with open('log.log', 'a') as f:
            f.write(f'\n{i_0}')
        # except:
        #     print(f'failed at {i_0}')
    