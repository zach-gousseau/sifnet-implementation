import xarray as xr
import glob
import re
import os
import dask
import numpy as np

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371e3 # Radius of earth in kilometers. 
    return c * r

def repeat_1d(arr, n):
    return np.repeat(arr[np.newaxis, :], n, axis=0)

def create_distance_grid(ds):
    """
    # User Haversine to create a "distances" grid
    # Each cell i is given the distance between cell i+1 and cell i-1
    # Done in both East and North directions
    """

    lats = repeat_1d(ds.latitude.values, ds.longitude.shape[0]).T
    lons = repeat_1d(ds.longitude.values, ds.latitude.shape[0])

    assert lats.shape == lons.shape

    distx = haversine(lats[1:-1, 1:-1], lons[1:-1, 2:], lats[1:-1, 1:-1], lons[1:-1, :-2])
    disty = haversine(lats[2:, 1:-1], lons[1:-1, 1:-1], lats[:-2, 1:-1], lons[1:-1, 1:-1])

    distx = np.pad(distx, ((1, 1), (1, 1)), 'constant', constant_values=np.nan)
    disty = np.pad(disty, ((1, 1), (1, 1)), 'constant', constant_values=np.nan)

    areas = (distx / 2) * (disty / 2)

    ds = ds.assign_coords(dict(
        distx=(('latitude', 'longitude'), distx),
        disty=(('latitude', 'longitude'), disty),
        area=(('latitude', 'longitude'), areas),
    ))
    return ds

def invert_lagged_cumsum(arr, initial_two, axis=0):
    arr = np.append(initial_two, arr, axis=axis)

    # Temporary arrays containing the cumsum using every 2nd item starting at index 0 and 1 respectively 
    arr0 = np.cumsum(arr[0::2], axis=axis)  
    arr1 = np.cumsum(arr[1::2], axis=axis)
    
    # Re-combine by "zipping"
    out = np.empty_like(arr, dtype=arr0.dtype)
    out[0::2] = arr0
    out[1::2] = arr1
    return out 

def repeat(arr, n):
    return np.repeat(arr[np.newaxis, :, :], n, axis=0)

def advection(U, V, Vol, Gx, Gy):
    # Gy = repeat(Gy, len(Vol))
    # Gx = repeat(Gx, len(Vol))

    adv = (
        -V[:, 1:-1, 1:-1] * ((Vol[:, 1:-1, 2:] - Vol[:, 1:-1, :-2])/(2*Gy[1:-1, 1:-1])) + \
        -U[:, 1:-1, 1:-1] * ((Vol[:, 2:, 1:-1] - Vol[:, :-2, 1:-1])/(2*Gx[1:-1, 1:-1]))
    )
    return np.pad(adv, ((0, 0), (1,1), (1,1)), 'constant', constant_values=np.nan)

def divergence(U, V, Vol, Gx, Gy):
    # Gy = repeat(Gy, len(Vol))
    # Gx = repeat(Gx, len(Vol))
    
    div = (
        -Vol[:, 1:-1, 1:-1] * ((V[:, 1:-1, 2:] - V[:, 1:-1, :-2])/(2*Gy[1:-1, 1:-1])) + \
        -Vol[:, 1:-1, 1:-1] * ((U[:, 2:, 1:-1] - U[:, :-2, 1:-1])/(2*Gx[1:-1, 1:-1]))
    )
    return np.pad(div, ((0, 0), (1,1),(1,1)), 'constant', constant_values=np.nan)

def intensification(Vol, nsecs, difference='forward'):
    if difference == 'forward':
        inn = (Vol[1:, :, :] - Vol[:-1, :, :]) / (1 * nsecs)
        inn = np.pad(inn, ((0, 1), (0,0),(0,0)), 'constant', constant_values=np.nan)
    elif difference == 'central':
        inn = (Vol[2:, :, :] - Vol[:-2, :, :]) / (2 * nsecs)
        inn = np.pad(inn, ((1, 1), (0,0),(0,0)), 'constant', constant_values=np.nan)
    else:
        raise ValueError()
    return inn

def revert_intensification(inn, initial_vol, nsecs, difference='forward'):
    if difference == 'forward':
        assert len(initial_vol) == 1
        arr = inn[:-1]
        arr = arr * (1 * nsecs)
        arr = np.append(initial_vol, arr, axis=0)
        arr = np.cumsum(arr, axis=0) 
    elif difference == 'central':
        assert len(initial_vol) == 2
        arr = inn[1:-1]
        arr = invert_lagged_cumsum(arr * (2 * nsecs), initial_vol)
    else:
        raise ValueError()
    return arr

def calculate_all(ds, landmask, suffix='', use_vol=True):
    siconc = np.nan_to_num(ds.siconc.values)
    vsi = np.nan_to_num(ds.vsi.values)
    usi = np.nan_to_num(ds.usi.values)
    
    # Replace 0s with NaNs where it does not intersect the landmask
    landmask = np.broadcast_to(landmask, siconc.shape)
    
    siconc[~landmask] = np.nan
    vsi[~landmask] = np.nan
    usi [~landmask] = np.nan

    if use_vol:
        sithick = np.nan_to_num(ds.sithick.values)
        sithick[~landmask] = np.nan
        arr = siconc * sithick
    else:
        arr = siconc

    gx = ds.distx.values
    gy = ds.disty.values

    adv = advection(usi, vsi, arr, gx, gy)  # Advection
    div = divergence(usi, vsi, arr, gx, gy)  # Divergence
    inn = intensification(arr, 86400)  # Intensification

    res = inn - adv - div  # Residual

    ds['div' + suffix] = (('time', 'latitude', 'longitude'), div)
    ds['adv' + suffix] = (('time', 'latitude', 'longitude'), adv)
    ds['inn' + suffix] = (('time', 'latitude', 'longitude'), inn)
    ds['res' + suffix] = (('time', 'latitude', 'longitude'), res)
    return ds


if __name__ == '__main__':
    ncs = natural_sort(glob.glob('/home/zgoussea/scratch/glorys12/*.nc'))

    outpath = '/home/zgoussea/scratch/glorys12/glorys12_v2.zarr'

    with open('log.log', 'r') as f:
        last_write = int(f.readlines()[-1].strip())

    ds = xr.open_mfdataset(ncs)
    ds = ds.rio.write_crs(4326)

    ds = create_distance_grid(ds)
    ds['time'] = np.array([dt - 12 * 3600000000000 for dt in ds.time.values])

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

            ds_sub = calculate_all(ds_sub)
            print('calculated')
            ds_sub = ds_sub.isel(time=slice(1, -1))  # Remove first and last
            ds_sub = ds_sub.chunk({'longitude': 1080, 'latitude': -1})
            for var_ in ds_sub.data_vars:
                try:
                    del ds_sub[var_].attrs['grid_mapping']
                except KeyError:
                    pass

            if os.path.exists(outpath):
                ds_sub.to_zarr(outpath, append_dim='time')
            else:
                ds_sub.to_zarr(outpath)
        i_0 = i_1

        with open('log.log', 'a') as f:
            f.write(f'\n{i_0}')
        # except:
        #     print(f'failed at {i_0}')

# if __name__ == '__main__':
#     ncs = natural_sort(glob.glob('/home/zgoussea/scratch/glorys12/*.nc'))

#     outpath = '/home/zgoussea/scratch/glorys12/glorys12.zarr'

#     with open('log.log', 'r') as f:
#         last_write = int(f.readlines()[-1].strip())

#     ds = xr.open_mfdataset(ncs)
#     ds = ds.rio.write_crs(4326)

#     ds = create_distance_grid(ds)
#     ds['time'] = np.array([dt - 12 * 3600000000000 for dt in ds.time.values])

#     ds = calculate_all(ds)

#     ds = ds.chunk({'longitude': 216, 'latitude': 37})

#     i_0 = 0
#     stepsize = 20

#     i_1 = i_0 + stepsize

#     if not os.path.exists(outpath):
#         ds.isel(time=slice(i_0, i_1)).to_zarr(outpath)
#     i_0 += stepsize

#     print('did first')

#     while i_1 < len(ds.time):
#         print(i_1)
#         i_1 = min(i_0 + stepsize, len(ds.time))

#         if last_write <= i_0:
#             i_0 = i_1
#             continue
#         try:
#             ds.isel(time=slice(i_0, i_1)).to_zarr(outpath)
#             i_0 = i_1
#             with open('log.log', 'a') as f:
#                 f.write(i_0)
#         except:
#             print(f'failed at {i_0}')


# if __name__ == '__main__':
#     ncs = natural_sort(glob.glob('/home/zgoussea/scratch/glorys12/*.nc'))

#     outname = '/home/zgoussea/scratch/glorys12/glorys12.zarr'

#     for nc in ncs:
#         print(f'Doing: {nc}')
#         if nc in converted_files:
#             print('Already done!')
#             continue

#         ds = xr.open_dataset(nc)
#         ds = ds.rio.write_crs(4326)

#         ds = create_distance_grid(ds)
#         ds['time'] = np.array([dt - 12 * 3600000000000 for dt in ds.time.values])

#         ds = calculate_all(ds)

#         ds = ds.chunk({'longitude': 216, 'latitude': 37})

#         if not os.path.exists(outname):
#             ds.to_zarr(outname, mode='w')
#         else:
#             ds.to_zarr(outname, mode='a', append_dim='time')

#         with open('log.log', 'a') as f:
#             f.write(f'\n{nc}')