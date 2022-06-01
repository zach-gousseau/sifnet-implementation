import xarray as xr
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from tqdm import tqdm
import seaborn as sns
import glob
from calendar import monthrange, month_name
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import accuracy_score

from data_generator import DataGen

results_dir = '/home/zgoussea/scratch/sifnet_results/9'

def masked_MSE(mask):
    def loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        sq_diff = tf.multiply(tf.math.squared_difference(y_pred, y_true), mask)
        return tf.reduce_mean(sq_diff)
    return loss

def masked_binary_crossentropy(mask):
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return binary_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
    return loss

def masked_accuracy(mask):
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return accuracy_score(y_true_masked, y_pred_masked)
    return loss

def get_landmask(ds):
    return np.logical_not(np.isnan(ds.sic.isel(time=0)))


def create_time_timestep_matrix(preds, timestep_in_days=1):
    """Create a 2d matrix of datetimes where each row is the next timestep"""
    time_2d = np.repeat(preds.time.values[np.newaxis,:], len(preds.timestep), 0)
    time_2d = time_2d + np.array([np.timedelta64(timestep_in_days * i,'D') for i in range(len(preds.timestep))])[:, None]
    return time_2d

# Read dataset & drop unnecessary variables
ds = xr.open_zarr('/home/zgoussea/scratch/era5_hb_daily.zarr')
ds = xr.merge([ds[['siconc']], ds.drop('siconc')])
ds = ds.rename({'siconc': 'sic'})

# Only these variables
vars = ['sic', 'sst', 't2m', 'sshf', 'u10', 'v10']
ds = ds[vars]

# Get landmask from SIC
landmask = get_landmask(ds)

ds = ds.assign_coords({'month': ds.time.dt.month})

# Accuracies by month
fns = glob.glob(os.path.join(results_dir, 'preds_*.nc'))
accuracies = None
for i, timestep in enumerate(tqdm(range(60))):

    preds_timestep = xr.concat([xr.open_dataset(fn).sel(timestep=timestep) for fn in fns], dim='time')
    preds_timestep = preds_timestep.assign_coords({'launch_date': preds_timestep.time})
    preds_timestep = preds_timestep.assign_coords({'time': preds_timestep.time.values + np.timedelta64(timestep,'D')})

    y_true_binary = ds[['sic']] > 0.15
    y_pred_binary = preds_timestep > 0.5
    preds_timestep = xr.merge([y_pred_binary, y_true_binary], join='left')

    acc = []
    months = []

    for month, preds_slice in preds_timestep.groupby('launch_date.month'):
        mask = np.repeat(np.array(landmask)[None,:], len(preds_slice.time), axis=0)
        acc.append(masked_accuracy(mask)(preds_slice.pred, preds_slice.sic))
        months.append(month)

    acc = xr.DataArray(acc, dims=('month'), coords=dict(month=months))

    if accuracies is None:
        accuracies = acc
    else:
        accuracies = xr.concat([accuracies, acc], dim='timestep')

sns.heatmap(accuracies.T, yticklabels=[month_name[i][:3] for i in months])#, vmin=0.8, vmax=1)
plt.savefig(os.path.join(results_dir, 'heatmap.png'))