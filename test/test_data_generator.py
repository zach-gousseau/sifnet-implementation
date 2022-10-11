import sys
import os
import datetime
import xarray as xr
import numpy as np
import logging
import matplotlib.pyplot as plt

sys.path.append(os.path.join(sys.path[0], '..'))

from data_generator import DataGen
from train_pinn import read_and_combine_glorys_era5

def batch_generator(arr, batch_size, overlap=0):
    i0 = 0

    while i0 + overlap != len(arr):
        i1 = min(i0 + batch_size + overlap, len(arr))
        yield arr[i0: i1]

        i0 = i1 - overlap

X_VARS = [
    'sivol', 'adv_v2', 'div_v2', 'res_v2', 'siconc', 'sithick', 'zos'
]

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

month=5
num_timesteps_input=3
num_timesteps_predict=4

num_training_years=1

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ds = read_and_combine_glorys_era5(
    era5='/home/zgoussea/scratch/era5_hb_daily.zarr',
    glorys='/home/zgoussea/scratch/glorys12/glorys12_v2.zarr',
    start_year=1993,
    end_year=1997,
    lat_range=(51, 70),  # Hudson Bay
    lon_range=(-95, -65),  # Hudson Bay
    coarsen=1,
)

data_gen = DataGen()

# Read data
ds = data_gen.get_data(
    ds=ds,
    add_add=False,
    X_vars=X_VARS,
    Y_vars=['sivol', 'adv_v2', 'div_v2', 'res_v2']
    )

logging.info('Got data.')

# Get landmask from SIC
data_gen.create_landmask_from_nans(ds, var_='zos')
landmask = data_gen.landmask

logging.info('Creating generator')

# Create dataset iterator
datasets = data_gen.get_generator(
    ds,
    month=5,
    num_timesteps_input=num_timesteps_input,
    num_timesteps_predict=num_timesteps_predict,
    binary_sic=False,
    num_training_years=num_training_years,
    normalize=True
    )

logging.info('Iterating generator')
data = next(datasets)
logging.info('Done iterating generator')

train_X_batches = batch_generator(data['train_X'], batch_size=8, overlap=2)
train_X_batch = next(train_X_batches)

for i, map_ in enumerate(train_X_batch[0, :, :, :, :]):
    num = map_.shape[-1]
    fig, axs = plt.subplots(int(np.ceil(num / 3)), 3, figsize=(12, np.ceil(num / 3) * 4))

    for j in range(num):
        axs.flatten()[j].imshow(map_[..., j])
        axs.flatten()[j].set_title(X_VARS[j])
    plt.savefig(f'/home/zgoussea/projects/def-ka3scott/zgoussea/sifnet/figs/map_{i}.png')
    plt.close()


for j in range(num):
    plt.imshow(train_X_batch[8, 0, :, :, j])
    plt.colorbar()
    plt.title(X_VARS[j])
    plt.savefig(f'/home/zgoussea/projects/def-ka3scott/zgoussea/sifnet/figs/map_{X_VARS[j]}.png')
    plt.close()

for j in range(num):
    print(X_VARS[j])
    print(f'{train_X_batch[..., j].flatten().mean():.1f}')
    print(f'{train_X_batch[..., j].flatten().std():.1f}')
    print('')