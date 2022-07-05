import packaging
import xarray as xr
import os
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import rioxarray
import datetime
import numpy as np
from scipy.ndimage import uniform_filter1d
# import cartopy.crs as ccrs
import math
from tqdm import tqdm
from scipy.spatial import distance
import matplotlib.pyplot as plt
import argparse
import datetime
import logging
import os
import sys
from calendar import monthrange

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow import keras
from tqdm import tqdm

from data_generator import DataGen
from losses import masked_accuracy, masked_binary_crossentropy, masked_MSE
from glorys12.calculate_adv import revert_intensification
from model import (
    spatial_feature_pyramid_net_hiddenstate_ND,
    spatial_feature_pyramid_net_vectorized_ND,
)

# GLORYS12 Copernicus Login Information 
USERNAME = 'zgousseau'
PASSWORD = os.environ.get('CMEMS_PASS')
DATASET_ID = 'cmems_mod_glo_phy_my_0.083_P1D-m'

# Model Parameters
NUM_TIMESTEPS_INPUT = 2
NUM_TIMESTEPS_PREDICT = 3
BINARY = True

# First number referring to initial training, second for subsequent training
EPOCHS = (
    200,
    5, #100,
)

BATCH_SIZE = 16
TRAINING_YEARS = 1

def batch_generator(arr, batch_size):
    i0 = 0

    while i0 < len(arr):
        i1 = min(i0 + batch_size, len(arr))
        yield arr[i0: i1]

        i0 = i1

def grad(model, inputs, targets, loss_function):
    with tf.GradientTape() as tape:
        loss_value = get_loss_value(model, inputs, targets, loss_function)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def get_loss_value(model, inputs, targets, loss_function):
    y_pred = model(inputs, training=True)
    loss_value = loss_function(targets, y_pred)
    return loss_value


def train(month, ds, save_path="", X_vars=None, predict_flux=True):

    data_gen = DataGen()

    Y_vars = ['adv', 'div', 'res', 'siconc'] if predict_flux else ['siconc'] 
    
    ds = data_gen.get_data(
        ds=ds,
        add_add=True,
        X_vars=X_vars,
        Y_vars=Y_vars
        )

    # Get landmask from SIC
    landmask = data_gen.get_landmask_from_nans(ds, var_='siconc')

    num_vars_to_predict = 3 if predict_flux else 1 

    # Loss function
    if BINARY:
        # mask = tf.expand_dims(
        #     np.transpose(
        #         np.repeat(np.array(landmask)[..., None], NUM_TIMESTEPS_PREDICT, axis=2),
        #         (2, 0, 1),
        #     ),
        #     axis=0,
        # )
        mask = np.broadcast_to(landmask, (num_vars_to_predict, NUM_TIMESTEPS_PREDICT, landmask.shape[0], landmask.shape[1]))
        mask = np.moveaxis(mask, 0, -1)

        loss_4d = masked_binary_crossentropy(mask=mask)
        loss_3d = masked_binary_crossentropy(mask=mask[..., 0])
    else:
        mask = np.expand_dims(landmask, [0, -1])
        loss = masked_MSE(mask=mask)

    preds = None
    loss_curves = None

    # Create dataset iterator
    datasets = data_gen.get_generator(
        ds,
        month=month,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_predict=NUM_TIMESTEPS_PREDICT,
        binary_sic=BINARY,
        num_training_years=TRAINING_YEARS,
    )

    # Get data dims
    image_size = len(ds.latitude), len(ds.longitude)
    num_vars = len(data_gen.X_vars)

    logging.info(f'Spatial dimensions: {image_size}')
    logging.info(f'Number of input variables: {num_vars}')

    # Create model & compile
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    model = spatial_feature_pyramid_net_hiddenstate_ND(
        input_shape=(NUM_TIMESTEPS_INPUT, *image_size, num_vars),
        output_steps=NUM_TIMESTEPS_PREDICT,
        l2=0,
        num_output_vars=num_vars_to_predict,
    )

    # model.compile(
    #     loss=loss, optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # )

    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # Begin loop over datasets
    logging.info("Begin training procedure.")
    i = 0
    for data in tqdm(datasets):

        # history = model.fit(
        #     data["train_X"],
        #     data["train_Y"],
        #     batch_size=BATCH_SIZE,
        #     epochs=EPOCHS[0] if i == 0 else EPOCHS[1],
        #     validation_data=(data["test_X"], data["test_Y"]),
        #     callbacks=[early_stopping],
        #     verbose=0,
        # )

        num_epochs = EPOCHS[0] if i == 0 else EPOCHS[1]

        siconc_loss_train = []
        siconc_loss_test = []

        for epoch in range(num_epochs):
            siconc_loss_train_epoch = tf.keras.metrics.Mean()
            siconc_loss_test_epoch = tf.keras.metrics.Mean()

            train_X_batches = batch_generator(data['train_X'], BATCH_SIZE)
            train_Y_batches = batch_generator(data['train_Y'], BATCH_SIZE)
            test_X_batches = batch_generator(data['test_X'], BATCH_SIZE)
            test_Y_batches = batch_generator(data['test_Y'], BATCH_SIZE)

            # Begin iteration over batches
            batch_i = 0
            try:
                while True:

                    train_X_batch = next(train_X_batches)
                    train_Y_batch = next(train_Y_batches)
                    test_X_batch = next(test_X_batches)
                    test_Y_batch = next(test_Y_batches)

                    # Remove siconc from Ys if predicting on fluxes, while saving siconc for loss calculation
                    # We want to keep siconc in the loop, but not use it for training, so that we can get predictions
                    # for siconc later. 
                    train_siconc = train_Y_batch[..., -1]
                    test_siconc = test_Y_batch[..., -1]
                    if predict_flux:
                        train_Y_batch = train_Y_batch[..., :-1]
                        test_Y_batch = test_Y_batch[..., :-1]

                    # Get losses and gradients from training
                    loss_value_train, grads = grad(model, train_X_batch, train_Y_batch, loss_4d)
                    loss_value_test = get_loss_value(model, test_X_batch, test_Y_batch, loss_4d)

                    # Apply gradients
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    # Update average loss for this epoch
                    # epoch_loss_train.update_state(loss_value_train)
                    # epoch_loss_test.update_state(loss_value_test)

                    if predict_flux:
                        y_pred_train = model(train_X_batch, training=True)
                        y_pred_test = model(test_X_batch, training=True)
                        inn_train = np.sum(y_pred_train, axis=-1)
                        inn_test = np.sum(y_pred_test, axis=-1)
                        train_siconc_pred = revert_intensification(inn=inn_train, initial_vol=train_siconc[:2])
                        test_siconc_pred = revert_intensification(inn=inn_test, initial_vol=test_siconc[:2])

                        siconc_loss_train_epoch.update_state(loss_3d(train_siconc, train_siconc_pred))
                        siconc_loss_test_epoch.update_state(loss_3d(test_siconc, test_siconc_pred))
                    else:
                        siconc_loss_train_epoch.update_state(loss_value_train)
                        siconc_loss_test_epoch.update_state(loss_value_test)


                    print(f'Batch {batch_i}', end='\r')
                    batch_i += 1

            except StopIteration:
                print(f'Epoch: {epoch} -- train_loss: {siconc_loss_train_epoch.result().numpy()} -- test_loss: {siconc_loss_test_epoch.result().numpy()}')

                siconc_loss_train.append(siconc_loss_train_epoch)
                siconc_loss_test.append(siconc_loss_test_epoch)

        # Get loss curve
        loss_curve = pd.DataFrame(
            {
                "iteration": [i] * num_epochs,
                "test_loss": siconc_loss_test,
                "train_loss": siconc_loss_train,
            }
        )

        # Get predictions on validation set
        preds_month_array = model.predict(data["valid_X"])
        preds_month = {}
        for var_i in range(preds_month_array.shape[-1]):

            var_i_name = Y_vars[var_i]

            preds_month[var_i_name] = xr.DataArray(
                preds_month_array[..., var_i],  # preds_month[..., 0],
                dims=("time", "timestep", "latitude", "longitude"),
                coords=dict(
                    time=data["dates_valid"],
                    timestep=range(NUM_TIMESTEPS_PREDICT), 
                    latitude=ds.latitude,
                    longitude=ds.longitude,
                ),
            )

        preds_month = xr.Dataset(preds_month)

        # Append loss / preds
        preds = xr.concat([preds, preds_month], dim="time") if preds is not None else preds_month
        loss_curves = loss_curves.append(loss_curve) if loss_curves is not None else loss_curve

        # Save this version of the model
        model.save(os.path.join(save_path, f"model_{month}_{i}"))

        i += 1

    # Turn to xr.Dataset
    # preds = preds.to_dataset(name="pred")
    # preds = preds.assign_coords(
    #     doy=(
    #         ("time"),
    #         [
    #             f"{m}-{d}"
    #             for m, d in zip(preds.time.dt.month.values, preds.time.dt.day.values)
    #         ],
    #     )
    # )

    # Save results
    preds.to_netcdf(os.path.join(save_path, f"preds_{month}.nc"), mode="w")
    model.save(os.path.join(save_path, f"model_{month}"))
    loss_curves.to_csv(os.path.join(save_path, f"loss_{month}.csv"), index=False)

    logging.info(f"Results written to {save_path}")


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Parameters --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("month", help="which month's model?")

    args = parser.parse_args()
    month = int(args.month)

    save_path = "/home/zgoussea/scratch/sifnet_results/test_results"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.info('Begin program.')

    # Read data -----------------------
    era5 = xr.open_zarr('/home/zgoussea/scratch/era5_hb_daily.zarr')

    # Load GLORYS & set hour to zero to match ERA5
    # glorys = read_glorys_from_cop().load()
    # glorys = create_distance_grid(glorys)
    # glorys['time'] = np.array([dt - 12 * 3600000000000 for dt in glorys.time.values])
    glorys = xr.open_zarr('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr')
    # glorys = xr.open_mfdataset([f'/home/zgoussea/scratch/glorys12/glorys12_{i}.nc' for i in np.arange(0, 1480, 20)])
    # glorys['time'] = np.array([dt - 12 * 3600000000000 for dt in glorys.time.values])

    # Hudson Bay 
    glorys = glorys.sel(latitude=slice(51, 70), longitude=slice(-95, -65))

    # Smaller temporal window for testing
    s, e = datetime.datetime(1993, 1, 1), datetime.datetime(1997, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys = glorys.sel(time=slice(s, e))

    logging.info('Read and sliced.')

    # Interpolate ERA5 to match GLORYS
    era5 = era5.interp(latitude=glorys['latitude'], longitude=glorys['longitude'])

    logging.info('Interpolated ERA5.')

    # Drop ERA5 SIC in favor of GLORYS12
    era5 = era5.drop('siconc')

    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")

    ds = ds.coarsen({'latitude': 5, 'longitude': 5}, boundary='trim').mean()

    # Train -----------------------------

    tf.random.set_seed(42)

    # Use multiple GPUs
    # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with mirrored_strategy.scope():
    #     train(month, data_path=data_path, save_path=save_path)

    logging.info('Start training.')
    train(
        month,
        ds=ds,
        save_path=save_path,
        X_vars=['siconc', 'div', 'adv', 'res', 't2m'],
        predict_flux=True)


