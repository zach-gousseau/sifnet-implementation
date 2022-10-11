import xarray as xr

lat_range=(51, 70),  # Hudson Bay
lon_range=(-95, -65),  # Hudson Bay

if __name__ == '__main__':

    era5 = xr.open_zarr(era5) if isinstance(era5, str) else era5

    glorys1 = xr.open_dataset('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr').isel(time=slice(1, None))
    glorys2 = xr.open_dataset('/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes.zarr')

    # Slice to spatial region of interest
    glorys1 = glorys1.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
    glorys2 = glorys2.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))

    # Only read years requested
    s, e = datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys1 = glorys1.sel(time=slice(s, e))
    glorys2 = glorys2.sel(time=slice(s, e))

    logging.debug('Read and sliced.')

    # glorys = xr.merge([glorys1, glorys2])
    glorys = xr.combine_by_coords([glorys1, glorys2], coords=['latitude', 'longitude', 'time'], join="exact")

    logging.debug('Combined GLORYS datasets')

    # Interpolate ERA5 to match GLORYS
    era5 = era5.interp(latitude=glorys['latitude'], longitude=glorys['longitude'])

    logging.debug('Interpolated ERA5.')

    # Drop ERA5 SIC in favor of GLORYS12
    era5 = era5.drop('siconc')

    # Get volume
    glorys['sivol'] = glorys.siconc * glorys.sithick

    ds = 
    u = ds.mean()
    std = ds.std()