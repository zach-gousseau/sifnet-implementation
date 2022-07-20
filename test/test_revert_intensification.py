import sys
import os
import datetime
from sklearn.preprocessing import StandardScaler
import xarray as xr
import matplotlib.pyplot as plt
import logging
import numpy as np
import time

sys.path.append(os.path.join(sys.path[0], '..'))

from data_generator import DataGen
from glorys12.calculate_adv import revert_intensification

month=5
num_timesteps_input=3
num_timesteps_predict=4
num_training_years=1

if __name__ == '__main__':
    
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Read data -----------------------
    glorys12 = xr.open_zarr('/home/zgoussea/scratch/glorys12/glorys12_v2.zarr')
    
    glorys12['sivol'] = glorys12.siconc * glorys12.sithick
    
    logging.info('Opened dataset')

    # Smaller area for testing
    glorys12 = glorys12.sel(latitude=slice(55, 60), longitude=slice(-90, -85))

    # Smaller temporal window for testing
    s, e = datetime.datetime(1993, 1, 1), datetime.datetime(1996, 1, 1)
    glorys12 = glorys12.sel(time=slice(s, e))
    
    logging.info('Sliced')
    
    data_gen = DataGen()
    
    # Read data
    ds = data_gen.get_data(
        ds=glorys12,
        add_add=False,
        X_vars=['adv', 'div', 'res', 'inn', 'sivol'],
        Y_vars=['adv', 'div', 'res', 'inn', 'sivol']
        )
    
    logging.info('Called get_data()')
    
    # Create dataset iterator
    datasets = data_gen.get_generator(
        ds,
        month=5,
        num_timesteps_input=2,
        num_timesteps_predict=14,
        binary_sic=False,
        num_training_years=1,
        normalize=False,
    )
    
    logging.info('Begin generating datasets')
    
    data = next(datasets)
    
    fluxes = data['valid_Y'][0, ..., :3]
    inn = data['valid_Y'][0, ..., -2]
    sivol = data['valid_Y'][0, ..., -1]
    
    inn0 = ds.inn.values[:10]
    adv0 = ds.adv.values[:10]
    res0 = ds.res.values[:10]
    div0 = ds.div.values[:10]
    sivol0 = ds.sivol.values[:10]
    
    # Sum to get intensification
    inn_calc = np.sum(fluxes, axis=-1)
    inn_calc0 = np.sum([adv0, res0, div0])
    
    # Revert intensification to get sivol and remove initial volumes since they are ground truth
    sivol_pred = revert_intensification(inn=inn_calc[1:-1], initial_vol=sivol[:2], nsecs=86400)
    sivol_pred0 = revert_intensification(inn=inn0[1:-1], initial_vol=sivol0[:2], nsecs=86400)
    
    sivol_pred == sivol