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
    
    ds = read_and_combine_glorys_era5(
        era5='/home/zgoussea/scratch/era5_hb_daily.zarr',
        glorys='/home/zgoussea/scratch/glorys12/glorys12_v2_fluxes_siconc_v2.zarr', # Or glorys12_v2_fluxes_v2.zarr if using SIV instead of SIC
        start_year=1993,
        end_year=2006,
        lat_range=(51, 70),  # Hudson Bay
        lon_range=(-95, -65),  # Hudson Bay
        coarsen=4,
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

    save_path = "/home/zgoussea/scratch/sifnet_results/compare_10years_30days_forward_diff_siconc"
    # save_path = None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    # First number referring to initial training, second for subsequent training
    epochs = (60, 20)

    # Train -------------------------
    for month in months:
        m = Model(
            month,
            predict_flux=predict_flux,
            num_timesteps_predict=30,
            num_timesteps_input=3,
            num_training_years=10,
            save_path=save_path,
            suffix=suffix
            )

        # Uncomment to use multiple GPUs (unsure if functional)
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
        # python -i main.py --month 0 --predict-fluxes 0 --suffix test