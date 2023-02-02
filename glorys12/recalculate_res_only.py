import xarray as xr
import glob
import re
import os
import dask
import numpy as np

def calculate_res(ds, suffix=''):
    inn = ds.inn_v3.values
    adv = ds.adv_v2.values
    div = ds.div_v2.values

    res = inn - adv - div

    ds['res' + suffix] = (('time', 'latitude', 'longitude'), res)
    return ds

if __name__ == '__main__':

    inpath = '/home/zgoussea/projects/def-ka3scott/zgoussea/glorys12_v2_with_fluxes.zarr'

    ds = xr.open_dataset(inpath)

    i_0 = 0
    stepsize = 10

    i_1 = -99

    while i_1 < len(ds.time):
        i_1 = min(i_0 + stepsize, len(ds.time))
        print(i_1)

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_sub = ds.isel(time=slice(max(i_0, 0), i_1))  # Pad indexes because first and last will be NaNs (intensification)
            ds_sub = calculate_res(ds_sub, suffix='_v3')
            
            ds_sub = ds_sub[['res_v3']]
            
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

            outpath = '/home/zgoussea/projects/def-ka3scott/zgoussea/glorys12_v2_with_fluxes_res_v3.zarr'
            if os.path.exists(outpath):
                ds_sub.to_zarr(outpath, append_dim='time')
            else:
                ds_sub.to_zarr(outpath)

        i_0 = i_1