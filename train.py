import os
import xarray as xr
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import argparse
import pandas as pd
import sys
from tqdm import tqdm
from calendar import monthrange
import logging

from model import spatial_feature_pyramid_net_hiddenstate_ND, spatial_feature_pyramid_net_vectorized_ND
from losses import masked_MSE, masked_binary_crossentropy, masked_accuracy
from data_generator import DataGen

NUM_TIMESTEPS_INPUT = 3
NUM_TIMESTEPS_PREDICT = 60
BINARY = True

EPOCHS = 250
BATCH_SIZE = 16
TRAINING_YEARS = 10

def train(month, data_path, save_path=''):

    data_gen = DataGen()

    ds = data_gen.get_data(data_path)

    # Get landmask from SIC
    landmask = data_gen.get_landmask(ds)

    # Loss function
    if BINARY:
        mask = tf.expand_dims(np.transpose(np.repeat(np.array(landmask)[...,None], NUM_TIMESTEPS_PREDICT, axis=2), (2, 0, 1)), axis=0)
        loss = masked_binary_crossentropy(mask=mask)
    else:
        mask = np.expand_dims(landmask, [0, -1])
        loss = masked_MSE(mask=mask)

    # Get data dims
    image_size = len(ds.latitude), len(ds.longitude)
    num_vars = len(ds.data_vars)

    preds = None
    loss_curves = None

    # Create dataset iterator
    datasets = data_gen.get_generator(
        ds,
        month=month,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_predict=NUM_TIMESTEPS_PREDICT,
        binary=BINARY,
        num_training_years=TRAINING_YEARS,
        )

    # Create model & compile
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = spatial_feature_pyramid_net_hiddenstate_ND(
        input_shape=(NUM_TIMESTEPS_INPUT, *image_size, num_vars),
        output_steps=NUM_TIMESTEPS_PREDICT,
        l2=0,
    )

    model.compile(loss=loss, optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)) #, run_eagerly=True)

    # Begin loop over datasets
    logging.info('Begin training procedure.')
    i = 0
    for data in tqdm(datasets):
        history = model.fit(
            data['train_X'],
            data['train_Y'],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(data['test_X'], data['test_Y']),
            callbacks=[early_stopping],
            verbose=0,
        )

        # Get loss curve
        loss_curve = pd.DataFrame({
            'iteration': [i] * len(history.history['val_loss']),
            'val_loss': history.history['val_loss'],
            'loss': history.history['loss']
        })

        # Get predictions on validation set
        preds_month = model.predict(data['valid_X'])
        preds_month = xr.DataArray(
            preds_month[..., 0],
            dims=('time', 'timestep', 'latitude', 'longitude'),
            coords=dict(
                time=data['dates_valid'],
                latitude=ds.latitude,
                longitude=ds.longitude,
                ))
        
        # Append loss / preds 
        preds = xr.concat([preds, preds_month], dim='time') if preds is not None else preds_month
        loss_curves = loss_curves.append(loss_curve) if loss_curves is not None else loss_curve

        # Save this version of the model
        model.save(os.path.join(save_path, f'model_{month}_{i}'))

        i += 1

    # Turn to xr.Dataset
    preds = preds.to_dataset(name='pred')
    preds = preds.assign_coords(
        doy=(('time'), [f'{m}-{d}' for m, d in zip(preds.time.dt.month.values, preds.time.dt.day.values)]
    ))

    # Save results 
    preds.to_netcdf(os.path.join(save_path, f'preds_{month}.nc'), mode='w')
    model.save(os.path.join(save_path, f'model_{month}'))
    loss_curves.to_csv(os.path.join(save_path, f'loss_{month}.csv'), index=False)

    logging.info(f'Results written to {save_path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("month", help="which month's model?")

    args = parser.parse_args()
    month = int(args.month)

    data_path = '/home/zgoussea/scratch/era5_hb_daily.zarr'
    save_path = '/home/zgoussea/scratch/sifnet_results/8'

    if  not os.path.exists(save_path):
        os.makedirs(save_path)

    tf.random.set_seed(42)

    # Use multiple GPUs
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with mirrored_strategy.scope():
        train(month, data_path=data_path, save_path=save_path)

