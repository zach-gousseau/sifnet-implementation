import logging
from calendar import monthrange

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from tensorflow import keras
from tqdm import tqdm

from data_generator import DataGen
from losses import masked_accuracy, masked_binary_crossentropy, masked_MSE
from model import (
    spatial_feature_pyramid_net_hiddenstate_ND,
    spatial_feature_pyramid_net_vectorized_ND,
)

print("Finished imports")
NUM_TIMESTEPS_INPUT = 3
NUM_TIMESTEPS_PREDICT = 4
BINARY = True

# First number referring to initial training, second for subsequent training
EPOCHS = (
    2,
    1,
)

BATCH_SIZE = 16
TRAINING_YEARS = 2


def train(month, data_path, save_path=""):

    data_gen = DataGen()

    ds = data_gen.get_data(data_path)

    # Get landmask from SIC
    landmask = data_gen.get_landmask(ds)

    # Loss function
    if BINARY:
        mask = tf.expand_dims(
            np.transpose(
                np.repeat(np.array(landmask)[..., None], NUM_TIMESTEPS_PREDICT, axis=2),
                (2, 0, 1),
            ),
            axis=0,
        )
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
        binary_sic=BINARY,
        num_training_years=TRAINING_YEARS,
    )

    print("Got generator")

    # Create model & compile
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    model = spatial_feature_pyramid_net_hiddenstate_ND(
        input_shape=(NUM_TIMESTEPS_INPUT, *image_size, num_vars),
        output_steps=NUM_TIMESTEPS_PREDICT,
        l2=0,
    )

    model.compile(
        loss=loss, optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    )  # , run_eagerly=True)

    # Begin loop over datasets
    logging.info("Begin training procedure.")
    i = 0
    for data in tqdm(datasets):
        history = model.fit(
            data["train_X"],
            data["train_Y"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS[0] if i == 0 else EPOCHS[1],
            validation_data=(data["test_X"], data["test_Y"]),
            callbacks=[early_stopping],
            verbose=1,
        )
        if i == 3:
            break


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    month = 1
    data_path = "/home/zgoussea/scratch/era5_hb_daily.zarr"

    tf.random.set_seed(42)
    print("start training")
    train(month, data_path=data_path, save_path="")
