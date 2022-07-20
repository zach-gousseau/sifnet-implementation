from train_pinn import *

import argparse
import time 
import os
import logging

if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start = time.time()
    
    ds = read_and_combine_glorys_era5(
        era5_path='/home/zgoussea/scratch/era5_hb_daily.zarr',
        glorys_path='/home/zgoussea/scratch/glorys12/glorys12_v2.zarr',
        start_year=1993,
        end_year=2006,
        lat_range=(51, 70),  # Hudson Bay
        lon_range=(-95, -65),  # Hudson Bay
        coarsen=4,
    )

    # Parameters --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--month")  # Month number
    parser.add_argument('--predict-fluxes')  # 1 if True, 0 otherwise
    parser.add_argument("--suffix")  # Nominal suffix to give to experiment

    args = vars(parser.parse_args())
    month = int(args['month'])
    predict_flux = bool(int(args['predict_fluxes']))
    suffix = args['suffix']

    # Directory ---------------------

    save_path = "/home/zgoussea/scratch/sifnet_results/compare"
    # save_path = None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    # First number referring to initial training, second for subsequent training
    epochs = (100, 60)

    # Train -------------------------

    m = Model(
        month,
        predict_flux=predict_flux,
        num_timesteps_predict=60,
        num_timesteps_input=3,
        num_training_years=10,
        save_path=save_path,
        suffix=suffix
        )
    
    
    # Use multiple GPUs
    # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with mirrored_strategy.scope():

    m.train(
        ds=ds,
        X_vars=X_VARS,
        epochs=epochs,
        save_example_maps=None,
        early_stop_patience=20,
        batch_size=8,
        )

    logging.info(f'Finished in {round((time.time() - start) / 60, 1)} minutes.')