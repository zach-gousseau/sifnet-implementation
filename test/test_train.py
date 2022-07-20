import argparse
import datetime
import logging
import os
import sys
from calendar import monthrange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow import keras
from tqdm import tqdm

from data_generator import DataGen
from losses import masked_accuracy, masked_binary_crossentropy, masked_MSE
from model import (spatial_feature_pyramid_net_hiddenstate_ND,
                   spatial_feature_pyramid_net_vectorized_ND)

NUM_TIMESTEPS_INPUT = 3
NUM_TIMESTEPS_PREDICT = 1
BINARY = True

EPOCHS = 120
BATCH_SIZE = 8
TRAINING_YEARS = 1


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, data, landmask):
        self.model = model
        self.data = data
        self.landmask = np.logical_not(landmask)

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict(data['valid_X'])
        true = data['valid_Y']
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        im1 = axs[0].imshow(np.ma.masked_where(self.landmask, preds[26, 0, :, :, 0]), vmin=0, vmax=1)
        im2 = axs[1].imshow(np.ma.masked_where(self.landmask, true[26, 0, :, :, 0]), vmin=0, vmax=1)
        plt.savefig(f'figs/predictions_by_epoch/{epoch}.png')
        plt.close()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logging.info('Finished imports. Why does xarray take so long?')

    data_gen = DataGen()

    month = 1
    data_path = '/home/zgoussea/scratch/era5_hb_daily.zarr'
    ds = data_gen.get_data(data_path)

    logging.info('Read data.')

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

    # Create dataset iterator
    datasets = data_gen.get_generator(
        ds,
        month=month,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_predict=NUM_TIMESTEPS_PREDICT,
        binary=BINARY,
        num_training_years=TRAINING_YEARS,
        )

    logging.info('Created data generator')

    # Create model & compile
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = spatial_feature_pyramid_net_hiddenstate_ND(
        input_shape=(NUM_TIMESTEPS_INPUT, *image_size, num_vars),
        output_steps=NUM_TIMESTEPS_PREDICT,
    )

    model.compile(loss=loss, optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)) #, run_eagerly=True)
    
    tf.random.set_seed(42)
    # Begin loop over datasets
    i = 0
    data = next(datasets)
    history = model.fit(
        x=data['train_X'],
        y=data['train_Y'],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data['test_X'], data['test_Y']),
        callbacks=[early_stopping, PredictionCallback(model, data, landmask)],
        verbose=1,
    )
