from pickle import FALSE
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
import time

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

# First number referring to initial training, second for subsequent training
EPOCHS = (
    60,
    40,
)

def batch_generator(arr, batch_size, overlap=0):
    i0 = 0

    while i0 + overlap != len(arr):
        i1 = min(i0 + batch_size + overlap, len(arr))
        yield arr[i0: i1]

        i0 = i1 - overlap



class Model:

    def __init__(self, month, predict_flux=True, num_timesteps_predict=90, num_timesteps_input=3, num_training_years=10, save_path='', suffix=None):
        
        self.predict_flux = predict_flux
        self.num_vars_to_predict = 3 if self.predict_flux else 1 
        self.num_timesteps_predict = num_timesteps_predict
        self.num_timesteps_input = num_timesteps_input
        self.num_training_years = num_training_years
        self.save_path = save_path
        self.suffix = suffix
        self.month = month

    def grad(self, model, inputs, targets, loss_function):
        with tf.GradientTape() as tape:
            loss_value = self.get_loss_value(model, inputs, targets, loss_function)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def get_loss_value(self, model, inputs, targets, loss_function):
        y_pred = model(inputs, training=True)
        loss_value = loss_function(targets, y_pred)
        return loss_value

    def create_loss_function(self, binary=True, mask=None):
        if binary:
            mask = np.broadcast_to(mask, (self.num_vars_to_predict, self.num_timesteps_predict, mask.shape[0], mask.shape[1]))
            mask = np.moveaxis(mask, 0, -1)

            loss_4d = masked_binary_crossentropy(mask=mask)
            loss_3d = masked_binary_crossentropy(mask=mask[..., 0])
        else:
            # mask = np.expand_dims(mask, [0, -1])
            # loss = masked_MSE(mask=mask)
            raise NotImplementedError('Need to implement continuous SIC training')

        return loss_3d, loss_4d


    def train(self, ds, X_vars=None, batch_size=16, save_example_maps=False, early_stop_patience=5):

        data_gen = DataGen()

        Y_vars = ['adv', 'div', 'res', 'sivol'] if self.predict_flux else ['sivol'] 
        
        ds = data_gen.get_data(
            ds=ds,
            add_add=True,
            X_vars=X_vars,
            Y_vars=Y_vars
            )

        # Get landmask from SIC
        landmask = data_gen.get_landmask_from_nans(ds, var_='sivol')

        # Loss function
        loss_3d, loss_4d = self.create_loss_function(binary=True, mask=landmask)

        preds = None
        loss_curves = None

        # Create dataset iterator
        datasets = data_gen.get_generator(
            ds,
            month=self.month,
            num_timesteps_input=self.num_timesteps_input,
            num_timesteps_predict=self.num_timesteps_predict,
            binary_sic=True,
            num_training_years=self.num_training_years,
        )

        # Get data dims
        image_size = len(ds.latitude), len(ds.longitude)
        num_vars = len(data_gen.X_vars)

        logging.info(f'Spatial dimensions: {image_size}')
        logging.info(f'Number of input variables: {num_vars}')

        # Create model & compile
        model = spatial_feature_pyramid_net_hiddenstate_ND(
            input_shape=(self.num_timesteps_input, *image_size, num_vars),
            output_steps=self.num_timesteps_predict,
            l2=0,
            num_output_vars=self.num_vars_to_predict,
        )

        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        # Begin loop over datasets
        logging.info("Begin training procedure.")
        i = 0
        for data in datasets:
            iter_start = time.time()

            num_epochs = EPOCHS[0] if i == 0 else EPOCHS[1]

            sivol_loss_train = []
            sivol_loss_test = []

            curr_best = np.inf
            epochs_since_improvement = 0

            for epoch in range(num_epochs):
                sivol_loss_train_epoch = tf.keras.metrics.Mean()
                sivol_loss_test_epoch = tf.keras.metrics.Mean()

                batch_overlap = 2 if self.predict_flux else 0

                # Begin iteration over training batches
                train_X_batches = batch_generator(data['train_X'], batch_size, batch_overlap)
                train_Y_batches = batch_generator(data['train_Y'], batch_size, batch_overlap)

                logging.debug('Created training batch generators')

                batch_i = 0
                try:
                    while True:

                        train_X_batch = next(train_X_batches)
                        train_Y_batch = next(train_Y_batches)


                        # Remove sivol from Ys if predicting on fluxes, while saving sivol for loss calculation
                        # We want to keep sivol in the loop, but not use it for training, so that we can get predictions
                        # for sivol later. 
                        train_sivol = train_Y_batch[..., 0]
                        if self.predict_flux:
                            train_Y_batch = train_Y_batch[..., 1:]

                        # Also store first two readings if predicting on fluxes to use for inverting intensification
                        # calculation
                        if self.predict_flux:
                            train_X_batch_init = train_X_batch[:2]
                            train_sivol_init = train_sivol[:2]

                            train_X_batch = train_X_batch[2:]
                            train_Y_batch = train_Y_batch[2:]
                            train_sivol = train_sivol[2:]

                        # Get losses and gradients from training w.r.t predicted variable
                        loss_value_train, grads = self.grad(model, train_X_batch, train_Y_batch, loss_4d)

                        # Apply gradients
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        # Get loss w.r.t sivol if predicting fluxes else append previously calculated loss
                        if self.predict_flux:
                            y_pred_train = model(np.append(train_X_batch_init, train_X_batch, axis=0), training=True)
                            inn_train = np.sum(y_pred_train, axis=-1)
                            train_sivol_pred = revert_intensification(inn=inn_train, initial_vol=train_sivol_init)
                            train_sivol_pred = train_sivol_pred[2:]  # Remove initial volumes since they are ground truth

                            sivol_loss_train_epoch.update_state(loss_3d(train_sivol, train_sivol_pred))
                        else:
                            sivol_loss_train_epoch.update_state(loss_value_train)

                        print(f'Batch {batch_i}', end='\r')
                        batch_i += 1

                except StopIteration:

                    # Begin iteration over test batches
                    test_X_batches = batch_generator(data['test_X'], batch_size, batch_overlap)
                    test_Y_batches = batch_generator(data['test_Y'], batch_size, batch_overlap)

                    try:
                        while True:

                            test_X_batch = next(test_X_batches)
                            test_Y_batch = next(test_Y_batches)

                            # Remove sivol from Ys if predicting on fluxes, while saving sivol for loss calculation
                            # We want to keep sivol in the loop, but not use it for training, so that we can get predictions
                            # for sivol later. 
                            test_sivol = test_Y_batch[..., 0]
                            if self.predict_flux:
                                test_Y_batch = test_Y_batch[..., 1:]

                            # Also store first two readings if predicting on fluxes to use for inverting intensification
                            # calculation
                            if self.predict_flux:
                                test_X_batch_init = test_X_batch[:2]
                                test_sivol_init = test_sivol[:2]

                                test_X_batch = test_X_batch[2:]
                                test_Y_batch = test_Y_batch[2:]
                                test_sivol = test_sivol[2:]

                            # Get losses w.r.t sivol
                            if self.predict_flux:
                                y_pred_test = model(np.append(test_X_batch_init, test_X_batch, axis=0), training=True)
                                inn_test = np.sum(y_pred_test, axis=-1)
                                test_sivol_pred = revert_intensification(inn=inn_test, initial_vol=test_sivol_init)
                                test_sivol_pred = test_sivol_pred[2:]  # Remove initial volumes since they are ground truth

                                sivol_loss_test_epoch.update_state(loss_3d(test_sivol, test_sivol_pred))
                            else:
                                loss_value_test = self.get_loss_value(model, test_X_batch, test_Y_batch, loss_4d)
                                sivol_loss_test_epoch.update_state(loss_value_test)

                    except StopIteration:
                        pass

                    if save_example_maps is not None:
                        ex_preds = model.predict(test_X_batch)
                        true = test_Y_batch

                        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                        im1 = axs[0].imshow(np.ma.masked_where(~landmask, ex_preds[save_example_maps, 0, :, :, 0]), vmin=0, vmax=1)
                        im2 = axs[1].imshow(np.ma.masked_where(~landmask, true[save_example_maps, 0, :, :, 0]), vmin=0, vmax=1)

                        ex_preds_dir = os.path.join(self.save_path, 'predictions_by_epoch')
                        ex_preds_path = os.path.join(ex_preds_dir, f'e_{epoch}_ts_{save_example_maps}.png' if self.suffix is None else f'e_{epoch}_ts_{save_example_maps}_{self.suffix}.png')

                        if not os.path.exists(ex_preds_dir):
                            os.mkdir(ex_preds_dir)

                        plt.savefig(ex_preds_path)
                        plt.close()

                    sivol_loss_train_epoch = sivol_loss_train_epoch.result().numpy()
                    sivol_loss_test_epoch = sivol_loss_test_epoch.result().numpy()

                    print(f'Epoch: {epoch} -- train_loss: {sivol_loss_train_epoch} -- test_loss: {sivol_loss_test_epoch}')

                    sivol_loss_train.append(sivol_loss_train_epoch)
                    sivol_loss_test.append(sivol_loss_test_epoch)

                    # Early stopping criteria (TODO: make into its own class)
                    if sivol_loss_test_epoch < curr_best:
                        curr_best = sivol_loss_test_epoch
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

                    if epochs_since_improvement == early_stop_patience:
                        break


            # Get loss curve
            loss_curve = pd.DataFrame(
                {
                    "iteration": [i] * len(sivol_loss_test),
                    "test_loss": sivol_loss_test,
                    "train_loss": sivol_loss_train,
                }
            )

            # Get predictions on validation set (TODO: maybe batch this?)
            preds_month_array = model.predict(data["valid_X"])
            preds_month = {}
            for var_i in range(preds_month_array.shape[-1]):

                var_i_name = Y_vars[var_i]

                preds_month[var_i_name] = xr.DataArray(
                    preds_month_array[..., var_i],  # preds_month[..., 0],
                    dims=("time", "timestep", "latitude", "longitude"),
                    coords=dict(
                        time=data["dates_valid"],
                        timestep=range(self.num_timesteps_predict), 
                        latitude=ds.latitude,
                        longitude=ds.longitude,
                    ),
                )

            preds_month = xr.Dataset(preds_month)

            # Append loss / preds
            preds = xr.concat([preds, preds_month], dim="time") if preds is not None else preds_month
            loss_curves = loss_curves.append(loss_curve) if loss_curves is not None else loss_curve

            # Save this version of the model
            model.save(os.path.join(self.save_path, f"model_{self.month}_{i}"))

            logging.info(f'Finished {i}th iteration in {round((time.time() - iter_start) / 60, 1)} minutes.')

            i += 1

            break

        # Save results
        preds_path = os.path.join(self.save_path, f'preds_{self.month}.nc' if self.suffix is None else f'preds_{self.month}_{self.suffix}.nc')
        model_path = os.path.join(self.save_path, f'model_{self.month}' if self.suffix is None else f'model_{self.month}_{self.suffix}')
        loss_path = os.path.join(self.save_path, f'loss_{self.month}' if self.suffix is None else f'loss_{self.month}_{self.suffix}.csv')

        preds.to_netcdf(preds_path, mode="w")
        model.save(model_path)
        loss_curves.to_csv(loss_path, index=False)

        logging.info(f"Results written to {self.save_path}")


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    start = time.time()

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
    glorys = xr.open_zarr('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr')

    # Hudson Bay 
    glorys = glorys.sel(latitude=slice(51, 70), longitude=slice(-95, -65))

    # Smaller temporal window for testing
    s, e = datetime.datetime(1993, 1, 1), datetime.datetime(2006, 1, 1)
    # s, e = datetime.datetime(1993, 1, 1), datetime.datetime(1996, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys = glorys.sel(time=slice(s, e))

    logging.debug('Read and sliced.')

    # Interpolate ERA5 to match GLORYS
    era5 = era5.interp(latitude=glorys['latitude'], longitude=glorys['longitude'])

    logging.debug('Interpolated ERA5.')

    # Drop ERA5 SIC in favor of GLORYS12
    era5 = era5.drop('siconc')

    # Get volume
    glorys['sivol'] = glorys.siconc * glorys.sithick

    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")

    ds = ds.coarsen({'latitude': 4, 'longitude': 4}, boundary='trim').mean()

    # Train -----------------------------

    tf.random.set_seed(42)

    # Use multiple GPUs
    # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with mirrored_strategy.scope():
    #     train(month, data_path=data_path, save_path=save_path)

    X_vars = [
        
        # Sea ice -----------------
        # 'siconc',
        'sivol',
        # 'sithick', 
        
        # Fluxes ------------------
        'adv', 'div', 'res',  # Must include if self.predict_flux
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
        # 'zos',  # Sea surface height

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

    m = Model(
        month,
        predict_flux=True,
        num_timesteps_predict=60,
        num_timesteps_input=3,
        num_training_years=10,
        save_path=save_path,
        suffix='fluxes'
        )

    m.train(
        ds=ds,
        X_vars=X_vars,
        save_example_maps=None,
        early_stop_patience=20,
        batch_size=16,
        )

    logging.info(f'Finished in {round((time.time() - start) / 60, 1)} minutes.')
