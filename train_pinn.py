from multiprocessing.sharedctypes import Value
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
import copy
import warnings
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

X_VARS = [
    
    # Sea ice -----------------
    # 'siconc',
    'sivol',
    # 'sithick', 
    
    # Fluxes ------------------
    'adv_v2', 'div_v2', 'res_v2',  # Must include if self.predict_flux
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

def batch_generator(arr, batch_size, overlap=0):
    i0 = 0

    while i0 + overlap != len(arr):
        i1 = min(i0 + batch_size + overlap, len(arr))
        yield arr[i0: i1]

        i0 = i1 - overlap



class Model:

    def __init__(self, month, predict_flux=True, num_timesteps_predict=90, num_timesteps_input=3, num_training_years=10, save_path=None, suffix=None):
        
        self.predict_flux = predict_flux
        self.num_vars_to_predict = 3 if self.predict_flux else 1 
        self.num_timesteps_predict = num_timesteps_predict
        self.num_timesteps_input = num_timesteps_input
        self.num_training_years = num_training_years
        self.save_path = save_path
        self.suffix = suffix
        self.month = month
        
        self.Y_vars = ['adv_v2', 'div_v2', 'res_v2'] if self.predict_flux else ['sivol']
        self.X_vars = []
        
        if self.save_path is None:
            warnings.warn('No save_path passed to constructor; results will not be saved!')

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
            mask = np.broadcast_to(mask, (self.num_vars_to_predict, self.num_timesteps_predict, mask.shape[0], mask.shape[1]))
            mask = np.moveaxis(mask, 0, -1)

            loss_4d = masked_MSE(mask=mask)
            loss_3d = masked_MSE(mask=mask[..., 0])

        return loss_3d, loss_4d
    
    def flux_to_sivol(self, y_pred, sivol_init, revert_norm=True, remove_init=True):
        
        # Revert normalization if desired
        if revert_norm:
            
            # First ensure the sample has the correct number of channels 
            assert len(self.Y_vars) == y_pred.shape[-1]
            
            for i, Y_var in enumerate(self.Y_vars):
                y_pred[..., i] = self.revert_norm(y_pred[..., i], Y_var)
        
        # Sum to get intensification
        inn_train = np.sum(y_pred, axis=-1)
        
        # Revert intensification to get sivol and remove initial volumes since they are ground truth
        sivol_pred = revert_intensification(inn=inn_train[1:-1], initial_vol=sivol_init, nsecs=86400)
        if remove_init:
            sivol_pred = sivol_pred[2:]
        return sivol_pred
    
    def revert_norm(self, arr, Y_var):
        """TODO: This should be a function in data_gen"""
        return arr * self.data_gen.std[Y_var].values + self.data_gen.u[Y_var].values


    def train(self, ds, X_vars=None, batch_size=16, epochs=(60, 40), save_example_maps=False, early_stop_patience=5, random_seed=42):
        
        tf.random.set_seed(random_seed)

        self.data_gen = DataGen()
        
        if X_vars[0] != 'sivol':
            raise ValueError('Re-order X_vars such that sivol is first. Otherwise need to re-write to allow non-sivol target.')

        self.X_vars = X_vars
        
        ds = self.data_gen.get_data(
            ds=ds,
            add_add=True,
            X_vars=self.X_vars,
            Y_vars=self.Y_vars + ['sivol'] if self.predict_flux else self.Y_vars
            )

        # Get landmask from ZOS
        self.data_gen.create_landmask_from_nans(ds, var_='zos')
        landmask = self.data_gen.landmask

        # Loss function
        loss_3d, loss_4d = self.create_loss_function(binary=False, mask=landmask)

        preds = None
        loss_curves = None

        # Create dataset iterator
        datasets = self.data_gen.get_generator(
            ds,
            month=self.month,
            num_timesteps_input=self.num_timesteps_input,
            num_timesteps_predict=self.num_timesteps_predict,
            binary_sic=False,
            num_training_years=self.num_training_years,
        )

        # Get data dims
        image_size = len(ds.latitude), len(ds.longitude)
        num_vars = len(self.data_gen.X_vars)

        logging.info(f'Spatial dimensions: {image_size}')
        logging.info(f'Number of input variables: {num_vars}')

        # Create model & compile
        model = spatial_feature_pyramid_net_hiddenstate_ND(
            input_shape=(self.num_timesteps_input, *image_size, num_vars),
            output_steps=self.num_timesteps_predict,
            l2=0.001,
            num_output_vars=self.num_vars_to_predict,
            sigmoid_out=False,
        )

        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        # Begin loop over datasets
        logging.info("Begin training procedure.")
        i = 0
        for data in datasets:
            iter_start = time.time()

            num_epochs = epochs[0] if i == 0 else epochs[1]

            model_loss_train = []
            model_loss_test = []
            sivol_loss_train = []
            sivol_loss_test = []

            curr_best = np.inf
            epochs_since_improvement = 0

            for epoch in range(num_epochs):
                model_loss_train_epoch = tf.keras.metrics.Mean()
                model_loss_test_epoch = tf.keras.metrics.Mean()
                
                # Model error on sivol (only used when predicting fluxes)
                sivol_loss_test_epoch = tf.keras.metrics.Mean()
                sivol_loss_train_epoch = tf.keras.metrics.Mean()

                batch_overlap = 2 if self.predict_flux else 0

                # Begin iteration over training batches
                train_X_batches = batch_generator(data['train_X'], batch_size, batch_overlap)
                train_Y_batches = batch_generator(data['train_Y'], batch_size, batch_overlap)

                batch_i = 0
                try:
                    while True:

                        train_X_batch = next(train_X_batches)
                        train_Y_batch = next(train_Y_batches)

                        # Remove sivol from Ys if predicting on fluxes, while saving sivol for loss calculation
                        # We want to keep sivol in the loop, but not use it for training, so that we can get predictions
                        # for sivol later. We also pre-emptively revert the normalization on this copy of sivol.
                        train_sivol = train_Y_batch[..., 0]
                        train_sivol = self.revert_norm(train_sivol, 'sivol')
                        if self.predict_flux:
                            train_Y_batch = train_Y_batch[..., 1:]

                        # Also store first two readings if predicting on fluxes to use for inverting intensification
                        # calculation
                        # if self.predict_flux:
                        #     train_X_batch_init = train_X_batch[:2]
                        #     train_sivol_init = train_sivol[:2]

                        #     train_X_batch = train_X_batch[2:]
                        #     train_Y_batch = train_Y_batch[2:]
                        #     train_sivol = train_sivol[2:]
                        

                        # Get losses and gradients from training w.r.t predicted variable
                        model_loss, grads = self.grad(model, train_X_batch, train_Y_batch, loss_4d)
                        model_loss_train_epoch.update_state(model_loss)

                        # Apply gradients
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        # Convert preds to sivol
                        if self.predict_flux:
                            
                            # Get predictions and convert from fluxes to sivol (and revert normalization)
                            y_pred_train = model(train_X_batch, training=True)
                            y_pred_train = y_pred_train.numpy()
                            
                            train_sivol_pred = np.empty((y_pred_train.shape[:-1]))
                            for sample_i, sample in enumerate(y_pred_train):
                                train_sivol_pred[sample_i] = self.flux_to_sivol(sample, sivol_init=train_sivol[sample_i, :2], revert_norm=True, remove_init=False)
                            
                        else:
                            
                            # Get predictions and revert normalization
                            y_pred_train = model(train_X_batch, training=True)
                            y_pred_train = y_pred_train[..., 0]
                            train_sivol_pred = self.revert_norm(y_pred_train, 'sivol')
                            
                        # Update loss
                        sivol_loss_train_epoch.update_state(loss_3d(train_sivol, train_sivol_pred))

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
                            # for sivol later. We also pre-emptively revert the normalization on this copy of sivol.
                            test_sivol = test_Y_batch[..., 0]
                            test_sivol = self.revert_norm(test_sivol, 'sivol')
                            if self.predict_flux:
                                test_Y_batch = test_Y_batch[..., 1:]

                            # Also store first two readings if predicting on fluxes to use for inverting intensification
                            # calculation
                            # if self.predict_flux:
                            #     test_X_batch_init = test_X_batch[:2]
                            #     test_sivol_init = test_sivol[:2]

                            #     test_X_batch = test_X_batch[2:]
                            #     test_Y_batch = test_Y_batch[2:]
                            #     test_sivol = test_sivol[2:]

                            model_loss = self.get_loss_value(model, test_X_batch, test_Y_batch, loss_4d)
                            model_loss_test_epoch.update_state(model_loss)
                            
                            # Convert preds to sivol
                            if self.predict_flux:
                                # Get predictions and convert from fluxes to sivol (and revert normalization)
                                y_pred_test = model(test_X_batch, training=True)
                                y_pred_test = y_pred_test.numpy()
                                
                                test_sivol_pred = np.empty((y_pred_test.shape[:-1]))
                                for sample_i, sample in enumerate(y_pred_test):
                                    test_sivol_pred[sample_i] = self.flux_to_sivol(sample, sivol_init=test_sivol[sample_i, :2], revert_norm=True, remove_init=False)

                            else:
                                # Get predictions and revert normalization
                                y_pred_test = model(test_X_batch, training=True)
                                y_pred_test = y_pred_test[..., 0]
                                test_sivol_pred = self.revert_norm(y_pred_test, 'sivol')
                                
                            # Update loss
                            sivol_loss_test_epoch.update_state(loss_3d(test_sivol, test_sivol_pred))

                    except StopIteration:
                        pass

                    # if save_example_maps is not None:
                    #     if self.save_path is None:
                    #         raise ValueError('If saving example images, must pass save_path to constructor!')
                    #     ex_preds = model.predict(test_X_batch)
                    #     true = test_Y_batch

                    #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    #     im1 = axs[0].imshow(np.ma.masked_where(~landmask, ex_preds[save_example_maps, 0, :, :, 0]), vmin=0, vmax=1)
                    #     im2 = axs[1].imshow(np.ma.masked_where(~landmask, true[save_example_maps, 0, :, :, 0]), vmin=0, vmax=1)

                    #     ex_preds_dir = os.path.join(self.save_path, 'predictions_by_epoch')
                    #     ex_preds_path = os.path.join(ex_preds_dir, f'e_{epoch}_ts_{save_example_maps}.png' if self.suffix is None else f'e_{epoch}_ts_{save_example_maps}_{self.suffix}.png')

                    #     if not os.path.exists(ex_preds_dir):
                    #         os.mkdir(ex_preds_dir)

                    #     plt.savefig(ex_preds_path)
                    #     plt.close()

                    model_loss_train_epoch = model_loss_train_epoch.result().numpy()
                    model_loss_test_epoch = model_loss_test_epoch.result().numpy()
                    sivol_loss_train_epoch = sivol_loss_train_epoch.result().numpy()
                    sivol_loss_test_epoch = sivol_loss_test_epoch.result().numpy()
                    
                    if np.isnan(model_loss_train_epoch) or np.isnan(model_loss_test_epoch):
                        raise ValueError('Encountered NaN loss.')

                    print(f'Epoch: {epoch} \t || train_loss (model): {model_loss_train_epoch:.5f} -- test_loss (model): {model_loss_test_epoch:.5f}' + \
                          f' || train_loss (sivol): {sivol_loss_train_epoch:.5f} -- test_loss (sivol): {sivol_loss_test_epoch:.5f}')

                    model_loss_train.append(model_loss_train_epoch)
                    model_loss_test.append(model_loss_test_epoch)
                    sivol_loss_train.append(sivol_loss_train_epoch)
                    sivol_loss_test.append(sivol_loss_test_epoch)

                    # Early stopping criteria (TODO: make into its own class)
                    if model_loss_test_epoch < curr_best:
                        curr_best = model_loss_test_epoch
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

                    if epochs_since_improvement == early_stop_patience:
                        break


            # Get loss curve
            loss_curve = pd.DataFrame(
                {
                    "iteration": [i] * len(model_loss_test),
                    "model_test_loss": model_loss_test,
                    "model_train_loss": model_loss_train,
                    "sivol_test_loss": sivol_loss_test,
                    "sivol_train_loss": sivol_loss_train,
                }
            )
            
            # Use entire array as "batch" (TODO: Refactor for repetitive code...)
            valid_X_batch = data["valid_X"]
            valid_Y_batch = data["valid_Y"]
            valid_sivol = valid_Y_batch[..., 0]
            valid_sivol = self.revert_norm(valid_sivol, 'sivol')
            
            if self.predict_flux:
                valid_Y_batch = valid_Y_batch[..., 1:]

            # Also store first two readings if predicting on fluxes to use for inverting intensification
            # calculation
            # if self.predict_flux:
                # valid_X_batch_init = valid_X_batch[:2]
                # valid_sivol_init = valid_sivol[:2]

                # valid_X_batch = valid_X_batch[2:]
                # valid_Y_batch = valid_Y_batch[2:]
                # valid_sivol = valid_sivol[2:]
                
            # Get predictions on validation set
            preds_month_array = model.predict(valid_X_batch)
            
            if np.all(np.isnan(preds_month_array)):
                warnings.warn('Predicted all NaNs! Continuing training.')
            
            preds_month = {}
            for var_i, Y_var in enumerate(self.Y_vars):

                preds_month[Y_var] = xr.DataArray(
                    self.revert_norm(preds_month_array[..., var_i], Y_var),  
                    dims=("time", "timestep", "latitude", "longitude"),
                    coords=dict(
                        time=data["dates_valid"],
                        timestep=range(self.num_timesteps_predict), 
                        latitude=ds.latitude,
                        longitude=ds.longitude,
                    ),
                )
            
            # Add y_true
            preds_month['sivol_true'] = xr.DataArray(
                valid_sivol,  
                dims=("time", "timestep", "latitude", "longitude"),
                coords=dict(
                    time=data["dates_valid"],
                    timestep=range(self.num_timesteps_predict), 
                    latitude=ds.latitude,
                    longitude=ds.longitude,
                ),
            )
            
            # TODO: This does not remove the initial maps passed to revert the intensification calculation
            # The reversion should use the two days PRIOR to the validation period, otherwise the scores will
            # be inflated !
            if self.predict_flux:
                
                preds_sivol = np.empty((preds_month_array.shape[:-1]))
                for sample_i, sample in enumerate(preds_month_array):
                    preds_sivol[sample_i] = self.flux_to_sivol(sample, sivol_init=valid_sivol[sample_i, :2], revert_norm=True, remove_init=False)
                
                preds_month['sivol'] = xr.DataArray(
                    preds_sivol,  
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
            if self.save_path is not None:
                model.save(os.path.join(self.save_path, f"model_{self.month}_{i}"))

            logging.info(f'Finished {i}th iteration in {round((time.time() - iter_start) / 60, 1)} minutes.')

            i += 1

            break

        # Save results
        if self.save_path is not None:
            preds_path = os.path.join(self.save_path, f'preds_{self.month}.nc' if self.suffix is None else f'preds_{self.month}_{self.suffix}.nc')
            model_path = os.path.join(self.save_path, f'model_{self.month}' if self.suffix is None else f'model_{self.month}_{self.suffix}')
            loss_path = os.path.join(self.save_path, f'loss_{self.month}' if self.suffix is None else f'loss_{self.month}_{self.suffix}.csv')

            preds.to_netcdf(preds_path, mode="w")
            model.save(model_path)
            loss_curves.to_csv(loss_path, index=False)

            logging.info(f'Process finished; results written to {self.save_path}')
        else:
            logging.info(f'Process finished; results not saved.')
            
        return preds, model, loss_curves
        
        
def read_and_combine_glorys_era5(era5, glorys, start_year=1993, end_year=2020, lat_range=(None, None), lon_range=(None, None), coarsen=1):
    logging.debug('Reading datasets')
    # Read data -----------------------
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

    logging.debug('Calculated sea ice volume.')

    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")

    if coarsen > 1:
        ds = ds.coarsen({'latitude': coarsen, 'longitude': coarsen}, boundary='trim').mean()
        
    return ds
