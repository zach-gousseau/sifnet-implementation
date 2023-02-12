import packaging
import xarray as xr
import os
import rioxarray
import datetime
import numpy as np
from scipy.ndimage import uniform_filter1d
import copy
import warnings

import math
from scipy.spatial import distance
import argparse
import datetime
import logging
import os
from calendar import monthrange
import time

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data_generator import DataGen
from losses import masked_accuracy, masked_binary_crossentropy, masked_MSE
from glorys12.calculate_adv import revert_intensification
from model import (
    spatial_feature_pyramid_net_hiddenstate_ND,
    spatial_feature_pyramid_net_vectorized_ND,
)

"""
Training a model using the three ice budget variables (advection/divergence/residual) as the target.

The terms can either be calculated using sea ice concentration, or sea ice volume (concentration * thickness). 
If using concentration, set ice_var to 'siconc', and the GLORYS flux terms are adv_siconc, div_siconc, res_siconc.
If using volume, set ice_var to 'sivol', and the GLORYS flux terms are adv_v2, div_v2, res_v3.

In this code, the advection/divergence/residual terms are referred to as "fluxes". 

"""


# Set the target variable, either concentration (SIC) or volume (SIV=SIC*SIT)
ice_var = 'siconc' # 'siconc' 'sivol'

# Choose the input variables 
# IMPORTANT: Uncomment the appropriate flux terms 
X_VARS = [
    
    # Sea ice -----------------
    'siconc',
    # 'sivol',
    # 'sithick', 
    
    # Fluxes ------------------
    # 'adv_v2', 'div_v2', 'res_v3',  # Must include if self.predict_flux == True and ice_var == 'sivol'
    'adv_siconc', 'div_siconc', 'res_siconc',  # Must include if self.predict_flux == True and ice_var == 'siconc'
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
    """Generate overlapping (or not) batches of an array along the 0th dimension"""
    i0 = 0

    while i0 + overlap != len(arr):
        i1 = min(i0 + batch_size + overlap, len(arr))
        yield arr[i0: i1]

        i0 = i1 - overlap



class Model:

    def __init__(self, month, ice_var='siconc', predict_flux=True, num_timesteps_predict=90, num_timesteps_input=3, num_training_years=10, save_path=None, suffix=None):
        """
        :param int month: Month of the year (1-12) of interest
        :param str ice_var: Which ice variable we are forecasting. Either 'siconc' or 'sivol'. 
        :param bool predict_flux: Whether to use the ice budget (flux) terms to forecast 
        :param int num_timesteps_predict: Number of output timesteps (days)
        :param int num_timesteps_input: Number of input timesteps (days)
        :param int num_training_years: Number of years to train the model
        :param str save_path: Path in which to save the model weights and results
        :param str suffix: Suffix to append to output files (for run identification)
        """
        
        self.predict_flux = predict_flux
        self.num_vars_to_predict = 3 if self.predict_flux else 1 
        self.num_timesteps_predict = num_timesteps_predict
        self.num_timesteps_input = num_timesteps_input
        self.num_training_years = num_training_years
        self.save_path = save_path
        self.suffix = suffix
        self.month = month
        self.ice_var = ice_var


        if ice_var == 'siconc':
            self.Y_vars = ['adv_siconc', 'div_siconc', 'res_siconc'] if self.predict_flux else [ice_var]
        elif ice_var == 'sivol':
            self.Y_vars = ['adv_v2', 'div_v2', 'res_v3'] if self.predict_flux else [ice_var]
        else:
            raise ValueError('ice_var should be either \'siconc\' or \'sivol\'!')


        self.X_vars = []
        
        if self.save_path is None:
            warnings.warn('No save_path passed to constructor; results will not be saved!')

    def grad(self, model, inputs, targets, loss_function):
        """Calculate loss and gradients"""
        with tf.GradientTape() as tape:
            loss_value = self.get_loss_value(model, inputs, targets, loss_function)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def get_loss_value(self, model, inputs, targets, loss_function):
        """Only calculate loss"""
        y_pred = model(inputs, training=True)
        loss_value = loss_function(targets, y_pred)
        return loss_value

    def create_loss_function(self, binary=True, mask=None):
        """Create the loss functions with the appropriate dimensionality. Uses BCE if we're forecasting binary ice presence, otherwise MSE for SIC."""
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
    
    def flux_to_ice(self, y_pred, ice_init, revert_norm=True, remove_init=True, difference='forward'):
        """
        Calculate SIC (or SIV) by summing the fluxes, and adding the result to the initial ice conditions, as:
        (delta)SIC = Adv. - Div. + Res. 
        SIC_t+1 = SIC_t + (delta)SIC

        (delta)SIC is also referred to as "intensification".

        :param np.ndarray y_pred: Advection/divergence/residual predictions.  -> I think [batch_size, 3, T, W, H]
        :param np.ndarray ice_init: Initial sea ice conditions. -> I think [batch_size, 1, T, W, H]
        :param bool revert_norm: Whether to revert the normalization such that the output can be interpreted as SIC/SIV in m2/m3
        :param bool remove_init: Whether to remove the initial ice conditions,
            i.e. if T=90, once we remove the initial conditions we only get 88 days of forecast (since the first two days come from the ground truth)
        :param str difference: Either 'forward' or 'central'. Refers to the differentiation technique used to calculate the flux terms. Keep as 'forward'.
        """

        ice_init = ice_init[:1] if difference == 'forward' else ice_init[:2]
        
        # Revert normalization if desired
        if revert_norm:
            
            # First ensure the sample has the correct number of channels 
            assert len(self.Y_vars) == y_pred.shape[-1]
            
            for i, Y_var in enumerate(self.Y_vars):
                y_pred[..., i] = self.revert_norm(y_pred[..., i], Y_var)
        
        # Sum to get intensification
        inn = np.sum(y_pred, axis=-1)
        
        # Revert intensification to get ice and remove initial volumes since they are ground truth
        ice = revert_intensification(inn=inn, initial_vol=ice_init, nsecs=86400, difference=difference)
        if remove_init:
            ice = ice[1:]  if difference == 'forward' else ice[:2]
        return ice
    
    def revert_norm(self, arr, Y_var):
        """TODO: This should be a function in data_gen"""
        return arr * self.data_gen.std[Y_var].values + self.data_gen.u[Y_var].values


    def train(self, ds, X_vars=None, batch_size=16, epochs=(60, 40), save_example_maps=False, early_stop_patience=5, random_seed=42):
        """
        Train a model for the specified epochs. 

        :param xr.DataSet ds: Dataset containing the input and target variables. 
        :param list X_vars: List of input variables.
        :param int batch_size: Batch size to use.
        :param tuple epochs: Number of epochs to use as a tuple.
            First integer is the number of epochs to use for initial training on N years
            Second integer is the number of epochs to use for re-training on every subsequent year
        :param str save_example_maps: Path in which to save example outputs. Not functional currently. Leave as False. 
        :param int early_stopping_patience: Number of epochs to wait before ending training if no improvement.
        :param int random_seed: Random seed. 
        """
        
        tf.random.set_seed(random_seed)

        self.data_gen = DataGen(self.ice_var)
        
        if X_vars[0] != self.ice_var:
            raise ValueError('Re-order X_vars such that ice is first. Otherwise need to re-write to allow non-ice target.')

        self.X_vars = X_vars
        
        logging.info(f'Getting data')

        ds = self.data_gen.get_data(
            ds=ds,
            add_add=True,
            X_vars=self.X_vars,
            Y_vars=self.Y_vars + [self.ice_var] if self.predict_flux else self.Y_vars
            )

        logging.info(f'Got data')

        # Get landmask from ZOS -- this is arbitrary and should be changed. 
        self.data_gen.create_landmask_from_nans(ds, var_='zos')
        landmask = self.data_gen.landmask

        # Loss function -- We need it broadcasted to 3D and 4D for different purposes. Janky, probably a better way around this. 
        loss_3d, loss_4d = self.create_loss_function(binary=False, mask=landmask)

        preds = None  # Predictions for output
        loss_curves = None  # Loss curves for output

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
            sigmoid_out=True,
        )

        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        # Begin loop over datasets
        # First dataset is initial training over N years, subsequent datasets are single years. 
        logging.info("Begin training procedure.")
        i = 0
        for data in datasets:
            logging.info("Generated dataset.")
            iter_start = time.time()

            num_epochs = epochs[0] if i == 0 else epochs[1]
            
            # Lists to be filled iteratively 
            model_loss_train = []
            model_loss_test = []
            ice_loss_train = []
            ice_loss_test = []

            # Variables for early stopping
            curr_best = np.inf
            epochs_since_improvement = 0

            # Begin iterating over epochs
            for epoch in range(num_epochs):
                model_loss_train_epoch = tf.keras.metrics.Mean()
                model_loss_test_epoch = tf.keras.metrics.Mean()
                
                # Model error on ice (only used when predicting fluxes)
                ice_loss_test_epoch = tf.keras.metrics.Mean()
                ice_loss_train_epoch = tf.keras.metrics.Mean()

                # If we're using fluxes, we need to overlap the batches by 2 since we need  
                # 2 days of initial conditions to calculate SIC from the fluxes. 
                batch_overlap = 2 if self.predict_flux else 0

                # Begin iteration over training batches
                train_X_batches = batch_generator(data['train_X'], batch_size, batch_overlap)
                train_Y_batches = batch_generator(data['train_Y'], batch_size, batch_overlap)

                batch_i = 0
                try:
                    while True:
                        
                        logging.info("Generating batch.")
                        train_X_batch = next(train_X_batches)
                        train_Y_batch = next(train_Y_batches)
                        logging.info("Generated batch.")


                        # Remove ice from Ys if predicting on fluxes, while saving ice for loss calculation
                        # We want to keep ice in the loop, but not use it for training, so that we can get predictions
                        # for ice later. We also pre-emptively revert the normalization on this copy of ice.
                        train_ice = train_Y_batch[..., 0]
                        train_ice = self.revert_norm(train_ice, self.ice_var)
                        if self.predict_flux:
                            train_Y_batch = train_Y_batch[..., 1:]
                        
                        # Get losses and gradients from training w.r.t predicted variable
                        logging.info("Training step.")
                        model_loss, grads = self.grad(model, train_X_batch, train_Y_batch, loss_4d)
                        model_loss_train_epoch.update_state(model_loss)

                        # Apply gradients
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                        # Convert preds to ice
                        if self.predict_flux:
                            
                            # Get predictions and convert from fluxes to ice (and revert normalization)
                            y_pred_train = model(train_X_batch, training=True)
                            y_pred_train = y_pred_train.numpy()
                            
                            train_ice_pred = np.empty((y_pred_train.shape[:-1]))
                            for sample_i, sample in enumerate(y_pred_train):
                                train_ice_pred[sample_i] = self.flux_to_ice(sample, ice_init=train_ice[sample_i, :2], revert_norm=True, remove_init=False)
                            
                        else:
                            
                            # Get predictions and revert normalization
                            y_pred_train = model(train_X_batch, training=True)
                            y_pred_train = y_pred_train[..., 0]
                            train_ice_pred = self.revert_norm(y_pred_train, self.ice_var)
                            
                        # Update loss
                        ice_loss_train_epoch.update_state(loss_3d(train_ice, train_ice_pred))

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

                            # Remove ice from Ys if predicting on fluxes, while saving ice for loss calculation
                            # We want to keep ice in the loop, but not use it for training, so that we can get predictions
                            # for ice later. We also pre-emptively revert the normalization on this copy of ice.
                            test_ice = test_Y_batch[..., 0]
                            test_ice = self.revert_norm(test_ice, self.ice_var)
                            if self.predict_flux:
                                test_Y_batch = test_Y_batch[..., 1:]

                            # Get loss value without calculating gradients
                            model_loss = self.get_loss_value(model, test_X_batch, test_Y_batch, loss_4d)
                            model_loss_test_epoch.update_state(model_loss)
                            
                            # Convert preds to ice
                            if self.predict_flux:
                                # Get predictions and convert from fluxes to ice (and revert normalization)
                                y_pred_test = model(test_X_batch, training=True)
                                y_pred_test = y_pred_test.numpy()
                                
                                test_ice_pred = np.empty((y_pred_test.shape[:-1]))
                                for sample_i, sample in enumerate(y_pred_test):
                                    test_ice_pred[sample_i] = self.flux_to_ice(sample, ice_init=test_ice[sample_i, :2], revert_norm=True, remove_init=False)

                            else:
                                # Get predictions and revert normalization
                                y_pred_test = model(test_X_batch, training=True)
                                y_pred_test = y_pred_test[..., 0]
                                test_ice_pred = self.revert_norm(y_pred_test, self.ice_var)
                                
                            # Update loss
                            ice_loss_test_epoch.update_state(loss_3d(test_ice, test_ice_pred))

                    except StopIteration:
                        pass
                    
                    # # Create example plots --- not functional currently.
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
                    ice_loss_train_epoch = ice_loss_train_epoch.result().numpy()
                    ice_loss_test_epoch = ice_loss_test_epoch.result().numpy()
                    
                    if np.isnan(model_loss_train_epoch) or np.isnan(model_loss_test_epoch):
                        raise ValueError('Encountered NaN loss.')

                    print(f'Epoch: {epoch} \t || train_loss (model): {model_loss_train_epoch:.5f} -- test_loss (model): {model_loss_test_epoch:.5f}' + \
                          f' || train_loss (ice): {ice_loss_train_epoch:.5f} -- test_loss (ice): {ice_loss_test_epoch:.5f}')

                    model_loss_train.append(model_loss_train_epoch)
                    model_loss_test.append(model_loss_test_epoch)
                    ice_loss_train.append(ice_loss_train_epoch)
                    ice_loss_test.append(ice_loss_test_epoch)

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
                    "ice_test_loss": ice_loss_test,
                    "ice_train_loss": ice_loss_train,
                }
            )
            
            # Use entire array as "batch" (TODO: Refactor for repetitive code...)
            valid_X_batch = data["valid_X"]
            valid_Y_batch = data["valid_Y"]
            valid_ice = valid_Y_batch[..., 0]
            valid_ice = self.revert_norm(valid_ice, self.ice_var)
            
            if self.predict_flux:
                valid_Y_batch = valid_Y_batch[..., 1:]
                
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
            preds_month['ice_true'] = xr.DataArray(
                valid_ice,  
                dims=("time", "timestep", "latitude", "longitude"),
                coords=dict(
                    time=data["dates_valid"],
                    timestep=range(self.num_timesteps_predict), 
                    latitude=ds.latitude,
                    longitude=ds.longitude,
                ),
            )
            
            # Convert predictions from numpy to xarray dataset. 
            # TODO: This does not remove the initial maps passed to revert the intensification calculation
            # The reversion should use the two days PRIOR to the validation period, otherwise the scores will
            # be inflated !
            if self.predict_flux:
                
                preds_ice = np.empty((preds_month_array.shape[:-1]))
                for sample_i, sample in enumerate(preds_month_array):
                    preds_ice[sample_i] = self.flux_to_ice(sample, ice_init=valid_ice[sample_i, :2], revert_norm=True, remove_init=False)
                
                preds_month[self.ice_var] = xr.DataArray(
                    preds_ice,  
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

            break  # Break after initial training. IMPORTANT: Remove this if we want to train over the entire dataset !

        # Save results. Predictions are netcdf files, model as a keras model, loss curves as csv. 
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
        
        
def read_and_combine_glorys_era5(era5, glorys, start_year=1993, end_year=2020, lat_range=(None, None), lon_range=(None, None), coarsen=1, ice_var='siconc'):
    """
    Read the ERA5 and GLORYS12 datasets and combine them into a single xarray dataset. 
    :param (str or xr.Dataset) era5: Path to ERA5 dataset (or a xr.Dataset itself)
    :param (str or xr.Dataset) glorys: Path to GLORYS dataset (or a xr.Dataset itself)
    :param int start_year: Start year of interest
    :param int end_year: End year of interest
    :param tuple(int, int) lat_range: Latitude range of the region of interest
    :param tuple(int, int) lon_range: Longitude range of the region of interest
    :param int coarsen: Amount to coarsen the dataset 
        e.g. if coarsen=2 and the original dataset is 100*100, the coarsened dataset will be 50*50
        This is used to reduce computation time during prototyping. 
    :param str ice_var: Ice variable to use, either 'siconc' or 'sivol'. 
    """
    logging.debug('Reading datasets')
    # Read data -----------------------
    era5 = xr.open_zarr(era5) if isinstance(era5, str) else era5
    glorys = xr.open_zarr(glorys) if isinstance(glorys, str) else glorys  # Use this one -> '/home/zgoussea/scratch/glorys12/glorys12_v2_with_fluxes.zarr'

    # Slice to spatial region of interest
    glorys = glorys.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))

    # Only read years requested
    s, e = datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys = glorys.sel(time=slice(s, e))

    logging.debug('Read and sliced.')

    # Interpolate ERA5 to match GLORYS
    era5 = era5.interp(latitude=glorys['latitude'], longitude=glorys['longitude'])

    logging.debug('Interpolated ERA5.')

    # Drop ERA5 SIC in favor of GLORYS12
    era5 = era5.drop('siconc')

    # Get volume if desired
    if ice_var == 'sivol':
        glorys['sivol'] = glorys.siconc * glorys.sithick
        logging.debug('Calculated sea ice volume.')

    logging.debug('Merging GLORYS and ERA5...')
    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")
    logging.debug('Merged GLORYS and ERA5.')

    # Resample the dataset by desired amount
    if coarsen > 1:
        logging.debug('Resampling...')
        ds = ds.coarsen({'latitude': coarsen, 'longitude': coarsen}, boundary='trim').mean()
        logging.debug('Resampled.')
        
    return ds
