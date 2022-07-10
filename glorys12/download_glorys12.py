import datetime
import os
import numpy as np
import packaging
import rioxarray
import xarray as xr
from pydap.cas.get_cookies import setup_session
from pydap.client import open_url

USERNAME = "zgousseau"
PASSWORD = os.environ.get("CMEMS_PASS")  # Set password with: export CMEMS_PASS=
DATASET_ID = "cmems_mod_glo_phy_my_0.083_P1D-m"

def copernicusmarine_datastore(dataset, username, password):
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    print(session.cookies.get_dict())
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))  
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    return data_store

if __name__ == '__main__':

    # data_store = copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD)

    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, USERNAME, PASSWORD)
    # print(session.cookies.get_dict())
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    url = f'https://my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'
    data_store = xr.backends.PydapDataStore(open_url(url, session=session, timeout=86400))  
    ds = xr.open_dataset(data_store)
    ds = ds.rio.write_crs(4326)

    # print(ds)

    ds = ds.sel(latitude=slice(50, None)).isel(depth=0)

    # chunks = {'latitude': -1, 'longitude': 160}

    i_0 = 0
    stepsize = 20

    i_1 = i_0 + stepsize

    outpath = f'/home/zgoussea/scratch/glorys12/glorys12_{i_0}.nc'
    if not os.path.exists(outpath):
        ds.isel(time=slice(i_0, i_1)).to_netcdf(outpath)
    i_0 += stepsize

    print('did first')

    while i_1 < len(ds.time):
        print(i_1)
        i_1 = min(i_0 + stepsize, len(ds.time))
        outpath = f'/home/zgoussea/scratch/glorys12/glorys12_{i_0}.nc'

        if os.path.exists(outpath):
            i_0 = i_1
            continue
        try:
            ds.isel(time=slice(i_0, i_1)).to_netcdf(outpath)
            i_0 = i_1
        except:
            print(f'failed at {i_0}')

