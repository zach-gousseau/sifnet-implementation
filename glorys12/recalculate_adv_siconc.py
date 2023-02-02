import xarray as xr
import glob
import re
import os
import dask
import numpy as np
import matplotlib.pyplot as plt

from calculate_adv import *

if __name__ == '__main__':

    zarrpath = '/home/zgoussea/projects/def-ka3scott/zgoussea/glorys12_v2_with_fluxes.zarr'

    ds = xr.open_dataset(zarrpath)
    landmask = ~np.isnan(ds.isel(time=0).zos.values)

    i_0 = 0
    stepsize = 10

    i_1 = -99

    while i_1 < len(ds.time):
        i_1 = min(i_0 + stepsize, len(ds.time))
        print(i_1)

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_sub = ds.isel(time=slice(max(i_0, 0), i_1 + 1))  # Pad indexes because first and last will be NaNs (intensification)

            ds_sub = calculate_all(ds_sub, landmask, suffix='_siconc', use_vol=False)
            print('calculated')
            
            if i_1 < len(ds.time):
                ds_sub = ds_sub.isel(time=slice(None, -1))  # Remove first and last
            
            ds_sub = ds_sub[['div_siconc', 'adv_siconc', 'inn_siconc', 'res_siconc']].drop(['depth'])
            
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
                
            outpath = '/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes_siconc_v2.zarr'
                
            if os.path.exists(outpath):
                ds_sub.to_zarr(outpath, append_dim='time')
            else:
                ds_sub.to_zarr(outpath)

        i_0 = i_1
