import packaging
import xarray as xr
import os
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import rioxarray
import datetime
import numpy as np
from scipy.ndimage import uniform_filter1d
import cartopy.crs as ccrs
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
from model import (
    spatial_feature_pyramid_net_hiddenstate_ND,
    spatial_feature_pyramid_net_vectorized_ND,
)

# GLORYS12 Copernicus Login Information 
USERNAME = 'zgousseau'
PASSWORD = os.environ.get('CMEMS_PASS')
DATASET_ID = 'cmems_mod_glo_phy_my_0.083_P1D-m'

# Model Parameters
NUM_TIMESTEPS_INPUT = 3
NUM_TIMESTEPS_PREDICT = 7
BINARY = True

# First number referring to initial training, second for subsequent training
EPOCHS = (
    200,
    100,
)

BATCH_SIZE = 16
TRAINING_YEARS = 1

def copernicusmarine_datastore(dataset, username, password):
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))  
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    return data_store


def read_glorys_from_cop():

    data_store = copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD)

    ds = xr.open_dataset(data_store)
    ds = ds.rio.write_crs(4326)

    return ds

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371e3 # Radius of earth in kilometers. 
    return c * r

def repeat_1d(arr, n):
    return np.repeat(arr[np.newaxis, :], n, axis=0)

def create_distance_grid(ds):
    """
    # User Haversine to create a "distances" grid
    # Each cell i is given the distance between cell i+1 and cell i-1
    # Done in both East and North directions
    """

    lats = repeat_1d(ds.latitude.values, ds.longitude.shape[0]).T
    lons = repeat_1d(ds.longitude.values, ds.latitude.shape[0])

    assert lats.shape == lons.shape

    distx = haversine(lats[1:-1, 1:-1], lons[1:-1, 2:], lats[1:-1, 1:-1], lons[1:-1, :-2])
    disty = haversine(lats[2:, 1:-1], lons[1:-1, 1:-1], lats[:-2, 1:-1], lons[1:-1, 1:-1])

    distx = np.pad(distx, ((1, 1), (1, 1)), 'constant', constant_values=np.nan)
    disty = np.pad(disty, ((1, 1), (1, 1)), 'constant', constant_values=np.nan)

    areas = (distx / 2) * (disty / 2)

    ds = ds.assign_coords(dict(
        distx=(('latitude', 'longitude'), distx),
        disty=(('latitude', 'longitude'), disty),
        area=(('latitude', 'longitude'), areas),
    ))
    return ds


def train(month, ds, save_path="", input_vars=None, output_vars=['siconc']):

    data_gen = DataGen()
    
    print(1)
    ds = data_gen.get_data(ds=ds)
    print(2)

    # Get landmask from SIC
    landmask = data_gen.get_landmask_from_nans(ds, var_='siconc')
    print(3)

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
        input_vars=input_vars,
        output_vars=output_vars
    )
    print(4)

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
    print(5)
    i = 0
    for data in tqdm(datasets):
        history = model.fit(
            data["train_X"],
            data["train_Y"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS[0] if i == 0 else EPOCHS[1],
            validation_data=(data["test_X"], data["test_Y"]),
            callbacks=[early_stopping],
            verbose=0,
        )

        # Get loss curve
        loss_curve = pd.DataFrame(
            {
                "iteration": [i] * len(history.history["val_loss"]),
                "val_loss": history.history["val_loss"],
                "loss": history.history["loss"],
            }
        )

        # Get predictions on validation set
        preds_month = model.predict(data["valid_X"])
        preds_month = xr.DataArray(
            preds_month[..., 0],
            dims=("time", "timestep", "latitude", "longitude"),
            coords=dict(
                time=data["dates_valid"],
                latitude=ds.latitude,
                longitude=ds.longitude,
            ),
        )

        # Append loss / preds
        preds = (
            xr.concat([preds, preds_month], dim="time")
            if preds is not None
            else preds_month
        )
        loss_curves = (
            loss_curves.append(loss_curve) if loss_curves is not None else loss_curve
        )

        # Save this version of the model
        model.save(os.path.join(save_path, f"model_{month}_{i}"))

        i += 1

    # Turn to xr.Dataset
    preds = preds.to_dataset(name="pred")
    preds = preds.assign_coords(
        doy=(
            ("time"),
            [
                f"{m}-{d}"
                for m, d in zip(preds.time.dt.month.values, preds.time.dt.day.values)
            ],
        )
    )

    # Save results
    preds.to_netcdf(os.path.join(save_path, f"preds_{month}.nc"), mode="w")
    model.save(os.path.join(save_path, f"model_{month}"))
    loss_curves.to_csv(os.path.join(save_path, f"loss_{month}.csv"), index=False)

    logging.info(f"Results written to {save_path}")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


month = 4




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


    # Read data -----------------------
    era5 = xr.open_zarr('/home/zgoussea/scratch/era5_hb_daily.zarr')

    # Load GLORYS & set hour to zero to match ERA5
    glorys = read_glorys_from_cop().load()
    glorys = create_distance_grid(glorys)
    glorys['time'] = np.array([dt - 12 * 3600000000000 for dt in glorys.time.values])

    # Hudson Bay 
    ds = glorys.sel(latitude=slice(51, 70), longitude=slice(-95, -65)).isel(depth=0)

    s, e = datetime.datetime(2000, 1, 1), datetime.datetime(2002, 1, 1)
    era5 = era5.sel(time=slice(s, e))
    glorys = glorys.sel(time=slice(s, e))

    # Interpolate ERA5 to match GLORYS
    era5 = era5.interp(latitude=glorys['latitude'], longitude=glorys['longitude'])

    # Drop ERA5 SIC
    era5 = era5.drop('siconc')

    ds = xr.combine_by_coords([era5, glorys], coords=['latitude', 'longitude', 'time'], join="inner")

    # Train -----------------------------

    tf.random.set_seed(42)

    # Use multiple GPUs
    # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with mirrored_strategy.scope():
    #     train(month, data_path=data_path, save_path=save_path)

    train(
        month,
        ds=ds,
        save_path=save_path,
        input_vars=None,
        output_vars=['siconc'])
