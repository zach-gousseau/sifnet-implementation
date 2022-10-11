from train_pinn import *

import argparse
import time 
import os
import logging

X_VARS = [
    
    # Sea ice -----------------
    # 'siconc',
    'sivol',
    # 'sithick', 
    
    # Fluxes ------------------
    # 'adv_v2', 'div_v2', 'res_v2',  # Must include if self.predict_flux
    # inn,

    # GLORYS ------------------
    # 'bottomT',  # Sea floor potential T
    # 'mlotst',  # Mixed layer thickness
    # 'so',  # Salinity
    # 'thetao',  # Sea water potential T
    # 'uo',  # Sea water velocity (E)
    # 'vo',  # Sea water velocity (N)
    # 'usi',  # Sea ice velocity (E)
    # 'vsi',  # Sea ice velocity (N)
    'zos',  # Sea surface height

    # ERA5 --------------------
    # 'd2m',  # 2m dewpoint
    # 'e',  # Evaporation
    # 'fg10',  # 10m wind gust since previous
    # 'msl',  # Mean sea level pressure
    # 'mwd',  # Mean wave direction
    # 'mwp',  # Mean wave period
    # 'sf',  # Snowfall
    # 'slhf',  # Surface latent heat flux
    # 'smlt',  # Snowmelt
    'sshf',  # Surface sensible heat flux
    # 'ssrd',  # Surface solar radiation downward
    'sst',  # Sea surface temperature
    # 'swh',  # Significant wave height
    't2m',  # 2m temperature
    # 'tcc',  # Total cloud cover
    # 'tp',  # Precip
    'u10',  # Wind speed (E)
    'v10'  # Wind speed (N)
]



if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start = time.time()

    
    # ds = read_and_combine_glorys_era5(
    #     era5='/home/zgoussea/scratch/era5_hb_daily.zarr',
    #     glorys='/home/zgoussea/scratch/glorys12/glorys12_v2.zarr',
    #     start_year=1993,
    #     end_year=2001,
    #     lat_range=(51, 70),  # Hudson Bay
    #     lon_range=(-95, -65),  # Hudson Bay
    #     coarsen=4,
    # )
    start_year=1993
    end_year=2001
    lat_range=(51, 70)  # Hudson Bay
    lon_range=(-95, -65)  # Hudson Bay
    
    # Read data -----------------------
    era5 = xr.open_zarr('/home/zgoussea/scratch/era5_hb_daily.zarr')
    glorys = xr.open_dataset('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr')

    # Slice to spatial region of interest
    glorys = glorys.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))

    # Only read years requested
    s, e = datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys = glorys.sel(time=slice(s, e))

    logging.debug('Read and sliced.')

    # Interpolate ERA5 to match GLORYS
    glorys = glorys.interp(latitude=era5['latitude'], longitude=era5['longitude'], method='nearest')

    logging.debug('Interpolated GLORYS.')

    era5 = era5.drop('siconc')
    # glorys = glorys.drop('siconc')

    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")

    ds['sivol'] = ds['siconc']

    month = 5
    predict_flux = False
    suffix = 'test_direct' if not predict_flux else 'test_fluxes'

    # Directory ---------------------
    
    save_path = "/home/zgoussea/scratch/sifnet_results/testing_as_sifnet_glorys"
    # save_path = None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    # First number referring to initial training, second for subsequent training
    epochs = (20, 1)

    # Train -------------------------

    m = Model(
        month,
        predict_flux=predict_flux,
        num_timesteps_predict=7,
        num_timesteps_input=3,
        num_training_years=5,
        save_path=save_path,
        suffix=suffix
        )

    preds, model, loss_curves = m.train(
        ds=ds,
        X_vars=X_VARS,
        epochs=epochs,
        save_example_maps=None,
        early_stop_patience=5,
        batch_size=8,
        )

    logging.info(f'Finished in {round((time.time() - start) / 60, 1)} minutes.')