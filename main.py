import netCDF4

from train_pinn import *

import argparse
import time 
import os
import logging
import sys

if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start = time.time()

    lat_range = (51, 70)
    lon_range = (-95, -65)

    start_year = 2000
    end_year = 2012

    glorys1 = xr.open_dataset('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr').isel(time=slice(1, None))
    glorys2 = xr.open_dataset('/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes_siconc_v2.zarr')

    # Slice to spatial region of interest
    glorys1 = glorys1.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
    glorys2 = glorys2.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))

    # Only read years requested
    s, e = datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 1, 1)
    glorys1 = glorys1.sel(time=slice(s, e))
    glorys2 = glorys2.sel(time=slice(s, e))

    glorys = xr.combine_by_coords([glorys1, glorys2], coords=['latitude', 'longitude', 'time'], join="exact")
    
    ds = read_and_combine_glorys_era5(
        era5='/home/zgoussea/scratch/era5_hb_daily.zarr',
        glorys=glorys,#'/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes_siconc_v2.zarr', # Or glorys12_v2_fluxes_v2.zarr if using SIV instead of SIC
        start_year=start_year,
        end_year=end_year,
        lat_range=lat_range,  # Hudson Bay
        lon_range=lon_range,  # Hudson Bay
        coarsen=0,
        cache_path='/home/zgoussea/scratch/sifnet_cache/'
    )

    # ds = read_and_combine_glorys_era5(
    #     era5='/home/zgoussea/scratch/era5_nwt_daily.zarr',
    #     glorys='/home/zgoussea/scratch/glorys12/glorys12_v2_with_fluxes.zarr',
    #     start_year=1993,
    #     end_year=2006,
    #     lat_range=(68, 77),  # Northwest Territories
    #     lon_range=(-140, -110),    # Northwest Territories
    #     coarsen=4,
    # )

    # Parameters --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--month")  # Month number
    parser.add_argument('--predict-fluxes')  # 1 if True, 0 otherwise
    parser.add_argument("--suffix")  # Nominal suffix to give to experiment

    args = vars(parser.parse_args())
    month = int(args['month'])
    predict_flux = bool(int(args['predict_fluxes']))
    suffix = args['suffix']

    if month == 0:
        months = range(1, 13)
    else:
        months = [month]

    # Directory ---------------------

    save_path = "/home/zgoussea/scratch/sifnet_results/tests_5_3_30"
    # save_path = None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    # First number referring to initial training, second for subsequent training
    epochs = (40, 20)

    # Train -------------------------
    for month in months:
        m = Model(
            month,
            predict_flux=predict_flux,
            num_timesteps_predict=30,
            num_timesteps_input=3,
            num_training_years=5,
            save_path=save_path,
            suffix=suffix
            )

        # # Uncomment to use multiple GPUs (unsure if functional)
        # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        # with mirrored_strategy.scope():

        m.train(
            ds=ds,
            X_vars=X_VARS,
            epochs=epochs,
            save_example_maps=None,
            early_stop_patience=5,
            batch_size=8,
            )

        logging.info(f'Finished month {month} in {round((time.time() - start) / 60, 1)} minutes.')

        # Example: 
        # python -i main.py --month 1 --predict-fluxes 1 --suffix test